import os
import json
import pandas as pd
from collections import Counter

# Build absolute path to JSON file
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
json_path = os.path.join(base_dir, "datasets", "stereoset_raw.json")

# Load the JSON
with open(json_path) as f:
    data = json.load(f)

rows = []
label_map = {
    "stereotype": 0,
    "anti-stereotype": 1,
    "unrelated": 2
}

# Go through all intrasentence examples
for group in data["data"]["intrasentence"]:
    context = group["context"]

    for sent_entry in group["sentences"]:
        sentence_text = sent_entry["sentence"]
        label_list = [lbl["label"] for lbl in sent_entry["labels"]]

        # Majority vote
        most_common_label, count = Counter(label_list).most_common(1)[0]
        if most_common_label not in label_map:
            continue

        rows.append({
            "context": context,
            "statement": sentence_text,
            "label": label_map[most_common_label],
            "label_name": most_common_label
        })

# Save to CSV
output_path = os.path.join(base_dir, "datasets", "stereoset.csv")
pd.DataFrame(rows).to_csv(output_path, index=False)

print("✅ Saved multiclass stereoset.csv with labels: stereotype (0), anti (1), unrelated (2)")