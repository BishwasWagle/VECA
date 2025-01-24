import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# # Load the dataset
df = pd.read_csv("vec_nodes_3.csv")

# # Encode node configurations into numeric format
X = df[['cpu', 'RAM', 'storage']].values


# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)

# Determine the optimal number of clusters using the Elbow method
ssd = []
k_range = range(1, 9)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    ssd.append(kmeans.inertia_)
print(ssd)

# Plot the Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, ssd, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Sum of Squared Distances (SSD)')
plt.xticks(k_range)
plt.grid()

# Highlight the Elbow point (k=3)
optimal_k = 4
plt.annotate(
    'Elbow point',
    xy=(optimal_k, ssd[optimal_k - 1]),
    xytext=(optimal_k + 1, ssd[optimal_k - 1] + 10),
    arrowprops=dict(facecolor='red', arrowstyle='->'),
    fontsize=10,
    color='red'
)
plt.legend()
plt.show()

# Train the k-means model with the optimal number of clusters (k=4 in this example)
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X_scaled)

# Assign cluster labels to the dataset
df['Cluster'] = kmeans.labels_

# Save the clustered dataset to a new CSV file
output_file_path = "clustered_vec_nodes.csv"
df.to_csv(output_file_path, index=False)

print(f"Clustered data saved to {output_file_path}")


