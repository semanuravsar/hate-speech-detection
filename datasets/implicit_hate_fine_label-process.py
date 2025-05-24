import pandas as pd
import os

# Paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_path = os.path.join(base_dir, "datasets", "implicit-hate-corpus", "implicit_hate_v1_stg2_posts.tsv")
output_path = os.path.join(base_dir, "datasets", "implicit_fine_labels.csv")

# Load and clean
df = pd.read_csv(input_path, sep="\t")
df.columns = df.columns.str.strip()
df = df[["post", "implicit_class"]].dropna()
df = df.rename(columns={"post": "text", "implicit_class": "label"})

# Map labels to integers
label_list = sorted(df["label"].unique())
label_map = {label: i for i, label in enumerate(label_list)}
df["label_id"] = df["label"].map(label_map)

# Save
df.to_csv(output_path, index=False)
print("✅ Saved to implicit_fine_labels.csv")
print("Class map:", label_map)