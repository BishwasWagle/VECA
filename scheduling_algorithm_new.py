import pandas as pd
import redis
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Load the clustered dataset
df = pd.read_csv("clustered_nodes_3.csv")

# Initialize Redis cache (assumes Redis server is running locally)
redis_client = redis.StrictRedis(host='localhost', port=6379, decode_responses=True)

# Mock RNN model for node availability prediction
def build_rnn_model():
    model = Sequential([
        Dense(16, input_dim=6, activation='relu'),  # Updated input_dim to 6
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')  # Output: availability probability
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Train a mock RNN model (for simplicity, this example skips actual training)
rnn_model = build_rnn_model()

# Functions for Algorithm 2
# def select_cluster(workflow, clusters):
#     # Select the cluster with capacity closest to the workflow's capacity
#     df_clustered = df.groupby("Cluster")[["cpu", "RAM", "storage"]].sum()
#     selected_cluster = df_clustered.loc[
#         (df_clustered["cpu"] >= workflow["cpu"]) &
#         (df_clustered["RAM"] >= workflow["RAM"]) &
#         (df_clustered["storage"] >= workflow["storage"])
#     ].idxmin()
#     return selected_cluster
def select_cluster(workflow, clusters):
    # Select the cluster with capacity closest to the workflow's capacity
    df_clustered = df.groupby("Cluster")[["cpu", "RAM", "storage"]].sum()
    
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
    
    return closest_cluster


def predict_node_availability(cluster, workflow):
    print(f"Selected Cluster: {cluster}")  # Debugging
    print(f"Type of cluster: {type(cluster)}, Value: {cluster}")

    # Perform the comparison for the selected cluster
    nodes = df[df["Cluster"] == cluster]
    
    if workflow.get("confidential_computing"):
        # Filter nodes supporting confidential computing (mocked for this example)
        nodes = nodes[
            (nodes["cpu"] >= workflow["cpu"]) &
            (nodes["RAM"] >= workflow["RAM"]) &
            (nodes["storage"] >= workflow["storage"])
        ]  # Adjust thresholds as needed
    
    availability_list = []
    for _, node in nodes.iterrows():
        # Prepare input for the RNN model
        input_features = np.array([[
            node["cpu"], node["RAM"], node["storage"],
            workflow["cpu"], workflow["RAM"], workflow["storage"]
        ]])  # Convert to NumPy array with the correct shape (1, 6)
        print(f"Input features for RNN: {input_features}")  # Debugging
        
        # Predict availability
        availability = rnn_model.predict(input_features)[0][0]  # Use the first batch and first output
        availability_list.append((node["nodeID"], availability))
    
    # Sort nodes by predicted availability in descending order
    ordered_nodes = sorted(availability_list, key=lambda x: x[1], reverse=True)
    
    # Store workflow and ordered nodes in Redis cache
    redis_client.set(f"workflow_{workflow['id']}_nodes", str(ordered_nodes))
    
    return ordered_nodes

def select_nearest_node(ordered_nodes):
    # Filter nodes with availability >= 0.8
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
        next_node = select_nearest_node(ordered_nodes)
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
    selected_cluster = select_cluster(workflow, df["Cluster"].unique())
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
    "cpu": 8,
    "RAM": 32,
    "storage": 1024,
    "confidential_computing": False
}

# Run the VEC Workflow Scheduler
vec_workflow_scheduler(workflow_example)
