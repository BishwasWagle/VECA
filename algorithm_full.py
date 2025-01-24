import pandas as pd
import redis
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

# Load the clustered dataset
df_cluster = pd.read_csv("clustered_vec_nodes.csv")
df_60_days = pd.read_csv("cleaned_node_availability_60_days.csv")

# Initialize Redis cache (assumes Redis server is running locally)
redis_client = redis.StrictRedis(host='localhost', port=6379, decode_responses=True)



# encoder = OneHotEncoder(sparse_output=False)
# encoded_weekday = encoder.fit_transform(df_cluster[['Weekday']])
# encoded_nodeID = encoder.fit_transform(df_cluster[['nodeID']])
# scaler = StandardScaler()
# normalized_hour = scaler.fit_transform(df_cluster[['Hour']])
# X = np.hstack([encoded_nodeID, encoded_weekday, normalized_hour])
# y = df_cluster['Availability'].values

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
# y_train, y_test = torch.tensor(y_train, dtype=torch.float32).view(-1, 1), torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Step 2: RNN Model
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

# input_size = X_train.shape[1]
# X_train = X_train.unsqueeze(1)
# X_test = X_test.unsqueeze(1)
# hidden_size = 128
# output_size = 1
# model = RNNModel(input_size, hidden_size, output_size)

# # Training
# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# epochs = 60
# for epoch in range(epochs):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train)
#     loss = criterion(outputs, y_train)
#     loss.backward()
#     optimizer.step()
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


def select_cluster(workflow, clusters):
    # Select the cluster with capacity closest to the workflow's capacity
    
    # df_clustered = df.groupby("Cluster")[["cpu", "RAM", "storage"]].sum()
    available_nodes = df_cluster[df_cluster["Availability"] == 1]
    df_clustered = available_nodes.groupby("Cluster")[["cpu", "RAM", "storage"]].sum()
    
    # Calculate the distance between workflow requirements and cluster capacities
    closest_cluster = None
    min_distance = float('inf')
    
    for cluster in df_clustered.index:
        cluster_capacity = df_clustered.loc[cluster]
        distance = np.sqrt(
            (cluster_capacity["cpu"] - workflow["cpu"])**2 +
            (cluster_capacity["RAM"] - workflow["RAM"])**2 +
            (cluster_capacity["storage"] - workflow["storage"])**2
        )
        if distance < min_distance:
            min_distance = distance
            closest_cluster = cluster
    print(closest_cluster, "^^^^^^^^^^^^^^^^")
    return closest_cluster


def predict_node_availability(cluster, workflow):
    print(f"Selected Cluster: {cluster}")  # Debugging
    print(f"Type of cluster: {type(cluster)}, Value: {cluster}")

    # Perform the comparison for the selected cluster
    nodes = df_60_days[df_60_days["Cluster"] == cluster]
    print(nodes["Availability"].value_counts())
    print(nodes.shape)
    
    # if workflow.get("confidential_computing"):
    #     # Filter nodes supporting confidential computing (mocked for this example)
    #     nodes = nodes[
    #         (nodes["cpu"] >= workflow["cpu"]) &
    #         (nodes["RAM"] >= workflow["RAM"]) &
    #         (nodes["storage"] >= workflow["storage"])
    #     ]  # Adjust thresholds as needed
    encoder = OneHotEncoder(sparse_output=False)
    encoded_weekday = encoder.fit_transform(df_60_days[['Weekday']])
    encoded_nodeID = encoder.fit_transform(df_60_days[['NodeID']])
    scaler = StandardScaler()
    normalized_hour = scaler.fit_transform(df_60_days[['Hour']])
    X = np.hstack([encoded_nodeID, encoded_weekday, normalized_hour])
    y = df_60_days['Availability'].values
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32).view(-1, 1), torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    availability_list = []
    input_size = X_train.shape[1]
    print(input_size)
    X_train = X_train.unsqueeze(1)
    X_test = X_test.unsqueeze(1)
    hidden_size = 128
    output_size = 1
    model = RNNModel(input_size, hidden_size, output_size)

    # Training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 60
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    model.eval()
    with torch.no_grad():
        print(X_test.shape, "X    test ###################################")
        predictions = model(X_test)
        predictions = (predictions >= 0.5).float()
        print(predictions[:20])
        accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
        print(f"Accuracy: {accuracy:.4f}") 
        

    node_ids = nodes["NodeID"].tolist()
    node_id_to_index = {node_id: idx for idx, node_id in enumerate(node_ids)}

    availability_list = []

    for i, node in nodes.iterrows():
        node_id = node["NodeID"]
        idx = node_id_to_index.get(node_id, None)

        if idx is not None:
            availability_list.append((node_id, predictions[idx].item()))
        else:
            print(f"Warning: NodeID {node_id} not found in predictions.")

    print(availability_list)
    # Sort nodes by predicted availability in descending order
    # print(availability_list)
    ordered_nodes = sorted(availability_list, key=lambda x: x[1], reverse=True)
    
    # Store workflow and ordered nodes in Redis cache
    redis_client.set(f"workflow_{workflow['id']}_nodes", str(ordered_nodes))
    
    return ordered_nodes

def select_nearest_node(ordered_nodes):
    # Filter nodes with availability >= 0.8
    print("******************************")
    # print(ordered_nodes)
    # eligible_nodes = [node for node in ordered_nodes if node[1] >= 0.8]
    # if eligible_nodes:
    #     return eligible_nodes[0][0]  # Return the top eligible node
    # else:
    #     return ordered_nodes[0][0]  # Fallback to the top node
    eligible_nodes = [node for node in ordered_nodes if node[1] >= 0.8]
    
    if eligible_nodes:
        return eligible_nodes[0][0]  # Return the top eligible node
    else:
        return ordered_nodes[0][0]  # Fallback to the top node

def execute_workflow(node, workflow):
    try:
        # Simulate workflow execution on the node
        print(f"Executing workflow {workflow['id']} on node {node}...")
        # Execution logic here (mocked for this example)
        return True
    except Exception as e:
        print(f"Execution failed on node {node}: {e}")
        # Retrieve ordered nodes from Redis cache
        ordered_nodes = eval(redis_client.get(f"workflow_{workflow['id']}_nodes"))
        # Select the next nearest node
        if ordered_nodes:
            next_node = select_nearest_node(ordered_nodes[1:])  # Exclude the failed node
            return execute_workflow(next_node, workflow)
        else:
            print("No available nodes for execution.")
            return False
            return execute_workflow(next_node, workflow)

def return_results(node, workflow):
    # Simulate collecting results from the node
    results = f"Results for workflow {workflow['id']} from node {node}"
    print(results)
    # Store results in a Flask server (mocked for this example)
    # Display results on the User UI (mocked for this example)
    return results

def vec_workflow_scheduler(workflow):
    # Step 1: Select the appropriate cluster
    selected_cluster = select_cluster(workflow, df_cluster["Cluster"].unique())
    # print(f"Selected Cluster: {selected_cluster}")
    
    # Step 2: Predict node availability in the selected cluster
    ordered_nodes = predict_node_availability(selected_cluster, workflow)
    
    # Step 3: Select the nearest node for execution
    execution_node = select_nearest_node(ordered_nodes)
    print(f"Selected Node for Execution: {execution_node}")
    
    # Step 4: Execute the workflow
    success = execute_workflow(execution_node, workflow)
    
    # Step 5: Return results
    if success:
        return return_results(execution_node, workflow)

# Example Workflow
workflow_example = {
    "id": "W1",
    "cpu": 32,
    "RAM": 128,
    "storage": 1024,
    "confidential_computing": False
}

# Run the VEC Workflow Scheduler
vec_workflow_scheduler(workflow_example)