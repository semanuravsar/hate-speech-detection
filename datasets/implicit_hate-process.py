import pandas as pd
import os

# Paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
input_path = os.path.join(base_dir, "datasets", "implicit-hate-corpus", "implicit_hate_v1_stg1_posts.tsv")
output_path = os.path.join(base_dir, "datasets", "latent_hatred_3class.csv")

# Load and clean
df = pd.read_csv(input_path, sep="\t")
df.columns = df.columns.str.strip()
df = df[["post", "class"]].dropna()
df = df.rename(columns={"post": "text", "class": "label"})

# Map to numeric classes
label_map = {
    "not_hate": 0,
    "implicit_hate": 1,
    "explicit_hate": 2
}
df["label_id"] = df["label"].map(label_map)

# Save
df.to_csv(output_path, index=False)
print("✅ Saved to latent_hatred_3class.csv")
print("Class map:", label_map)
