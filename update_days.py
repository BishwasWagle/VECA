import pandas as pd
import numpy as np

# Load the dataset
file_path = "cleaned_node_availability_60_days.csv"
df = pd.read_csv(file_path)

np.random.seed(42)

# Determine the number of rows
num_rows = len(df)

# Generate a random binary array with 20% zeros and 80% ones
availability = np.random.choice([0, 1], size=num_rows, p=[0.2, 0.8])

# Assign the generated availability to the column
df['Availability'] = availability

# Save the cleaned dataset
output_file_path = "cleaned_node_availability.csv"
df.to_csv(output_file_path, index=False)

print(f"Cleaned dataset saved at: {output_file_path}")