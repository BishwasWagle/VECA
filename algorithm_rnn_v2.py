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
from torch.utils.data import DataLoader, TensorDataset
import json


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


def predict_availability(cluster, workflow, device=None):
    # Set device
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Filter data for the given cluster
    nodes = df_60_days[df_60_days["Cluster"] == cluster]
    if nodes.empty:
        print(f"No nodes found for cluster {cluster}.")
        return []

    # Prepare features
    encoder_weekday = OneHotEncoder(sparse_output=False)
    encoder_nodeID = OneHotEncoder(sparse_output=False)
    scaler_hour = StandardScaler()

    # Encode features
    encoded_weekday = encoder_weekday.fit_transform(nodes[['Weekday']])
    encoded_nodeID = encoder_nodeID.fit_transform(nodes[['NodeID']])
    normalized_hour = scaler_hour.fit_transform(nodes[['Hour']])
    encoded_weekday_tensor = torch.tensor(encoded_weekday, dtype=torch.float32)
    encoded_nodeID_tensor = torch.tensor(encoded_nodeID, dtype=torch.float32)
    normalized_hour_tensor = torch.tensor(normalized_hour, dtype=torch.float32)

    # Combine features
    features = torch.cat((encoded_nodeID_tensor, encoded_weekday_tensor, normalized_hour_tensor), dim=1)


    # Target variable
    labels = torch.tensor(nodes['Availability'].values, dtype=torch.float32).view(-1, 1)

    # Train-test split
    split_idx = int(0.8 * len(features))
    X_train, X_test = features[:split_idx], features[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]

    # Move data to the specified device
    X_train, X_test = X_train.to(device), X_test.to(device)
    y_train, y_test = y_train.to(device), y_test.to(device)

    # Define model
    model = torch.nn.Sequential(
        torch.nn.Linear(X_train.shape[1], 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 1),
        torch.nn.Sigmoid()
    ).to(device)

    # Loss and optimizer
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # DataLoader for efficient training
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Train model
    epochs = 60
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

    # Evaluate model
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).round()
        predictions = (predictions >= 0.5).float()
        print(predictions)
        accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
        print(f"Accuracy: {accuracy:.4f}") 
        # precision = precision_score(y_test.cpu().numpy(), predictions.cpu().numpy())
        # recall = recall_score(y_test.cpu().numpy(), predictions.cpu().numpy())
        # f1 = f1_score(y_test.cpu().numpy(), predictions.cpu().numpy())
        # print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    # Prepare predictions for nodes
    node_id_columns = X_test[:, :encoded_nodeID.shape[1]]
    node_ids = encoder_nodeID.inverse_transform(node_id_columns.cpu().numpy())
    availability_list = []
    for idx, node_id in enumerate(node_ids):
        availability_list.append((node_id[0], predictions[idx].item()))
    
    print(availability_list)

    # Sort nodes by availability (descending) and store in Redis
    ordered_nodes = sorted(availability_list, key=lambda x: x[1], reverse=True)
    redis_client.set(f"workflow_{workflow['id']}_nodes", json.dumps(ordered_nodes))

    return ordered_nodes

def extract_node_ids_from_X_test(X_test, encoder):
    """
    Extracts NodeIDs from the one-hot encoded X_test tensor.

    Args:
        X_test (torch.Tensor): The input tensor containing one-hot encoded features.
        encoder (OneHotEncoder): The fitted OneHotEncoder for NodeID.

    Returns:
        List[int]: The decoded NodeIDs.
    """
    # Ensure X_test is in the correct shape
    if X_test.dim() == 3:
        X_test = X_test.squeeze(1)

    # Convert X_test to numpy for compatibility with encoder
    X_test_np = X_test.numpy()

    # Extract the columns corresponding to NodeID
    num_node_ids = len(encoder.categories_[0])  # Number of unique NodeIDs
    node_id_columns = X_test_np[:, :num_node_ids]  # Assuming NodeID is the first encoded feature

    # Decode the one-hot encoded NodeIDs
    node_ids = encoder.inverse_transform(node_id_columns)

    return node_ids.flatten().tolist()

def select_nearest_node(ordered_nodes):
    # Filter nodes with availability >= 0.8
    print("******************************")
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
    ordered_nodes = predict_availability(selected_cluster, workflow)
    
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