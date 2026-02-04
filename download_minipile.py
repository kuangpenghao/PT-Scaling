from datasets import load_dataset
import os

# Define dataset name and local path
dataset_name = "JeanKaddour/minipile"
local_path = "./local_datasets/minipile"

# Create directory if it doesn't exist
os.makedirs(local_path, exist_ok=True)

print(f"Downloading dataset {dataset_name}...")
# Load the dataset
dataset = load_dataset(dataset_name)

print(f"Saving dataset to {local_path}...")
# Save to disk
dataset.save_to_disk(local_path)

print("Done!")
