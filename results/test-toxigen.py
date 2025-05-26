import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import os
import sys

# 🧠 Adjust paths as needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # to access your model class if needed

from models.multitask_bert import MultiTaskBERT  # if you need to reconstruct the model class

# -------- CONFIG -------- #
CHECKPOINT_PATH = "results/baseline.pt"
TASK = "main"  # name of the task used in model.forward(task=...)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------------ #

print("📦 Loading ToxiGen dataset...")
toxigen = load_dataset("toxigen/toxigen-data", name="train", split="train")

print("🧠 Loading model checkpoint...")
model = MultiTaskBERT()
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

all_preds = []
all_labels = []

print("🔍 Running inference on ToxiGen...")
for i, row in enumerate(toxigen):
    text = row["generation"]
    label = int(row["prompt_label"])

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], task=TASK)
        pred = torch.argmax(logits, dim=1).item()
        pred = 1 if pred in [1, 2] else 0

    all_preds.append(pred)
    all_labels.append(label)

    if i % 10000 == 0:
        print(f"✅ Processed {i} samples")

print("\n📊 Evaluation on ToxiGen:")
print("Accuracy:", accuracy_score(all_labels, all_preds))
print("F1 Score:", f1_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=["non-toxic", "toxic"]))

print("\n📊 Evaluation on ToxiGen:")
print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
print(f"F1 Score:  {f1_score(all_labels, all_preds):.4f}")
