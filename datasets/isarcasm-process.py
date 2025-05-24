import pandas as pd
import os

# Base directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Input/output paths
input_path = os.path.join(base_dir, "datasets", "isarcasm_dataset.csv")
output_path = os.path.join(base_dir, "datasets", "isarcasm.csv")

# Load the CSV
df = pd.read_csv(input_path)
df.columns = df.columns.str.strip().str.lower()  # normalize headers

# Select and rename
df = df[["tweet", "sarcastic"]].dropna()
df = df.rename(columns={"tweet": "text", "sarcastic": "label"})
df["label"] = df["label"].astype(int)

# Save
df.to_csv(output_path, index=False)
print("✅ Saved to isarcasm.csv")
print(df["label"].value_counts())
