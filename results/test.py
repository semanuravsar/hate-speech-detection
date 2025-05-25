import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from models.multitask_bert import MultiTaskBERT
from scripts.utils import load_checkpoint
from scripts.dataset_loaders import LatentHatredDataset


def evaluate(model, dataloader, device, task_name="main"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, task=task_name)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"\n📈 Test set results for task '{task_name}':")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Load model ===
    model = MultiTaskBERT(dropout=0.1).to(device)
    checkpoint_path = "results/best_model_overall_lr2e-05_do0.1_mb32_ab8_mw1.0_st0_sa0_fi1.0.pt"
    load_checkpoint(model, path=checkpoint_path)

    # === Load test set ===
    test_dataset = LatentHatredDataset("datasets/latent_hatred_3class_test.csv")
    test_loader = DataLoader(test_dataset, batch_size=32)

    # === Evaluate ===
    evaluate(model, test_loader, device, task_name="main")


if __name__ == "__main__":
    main()
