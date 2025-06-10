import os, sys, time, argparse, itertools, random
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.append(os.path.expanduser("~/codes-v2"))

from models.multitask_bert import MultiTaskBERT
from scripts.dataset_loaders import (
    LatentHatredDataset, StereoSetDataset, ISarcasmDataset, ImplicitFineHateDataset
)
from scripts.utils import save_checkpoint, load_checkpoint

class BaseTextDataset(Dataset):
    def __init__(self, texts, labels):
        from scripts.utils import preprocess_text, encode_text
        self.texts = [preprocess_text(text) for text in texts]
        self.labels = labels
        self.encode_text = encode_text

    def __getitem__(self, idx):
        enc = self.encode_text(self.texts[idx])
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

@torch.no_grad()
def evaluate(model, dataloaders, device):
    model.eval()
    metrics = {}
    for task, loader in dataloaders.items():
        all_preds, all_labels = [], []
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask, task=task)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        metrics[task] = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds, average="macro"),
            "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
            "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        }
    return metrics

def train_one_epoch(model, dataloaders, optimizer, task_weights, device):
    model.train()
    total_loss = 0.0
    task_iters = {task: iter(loader) for task, loader in dataloaders.items()}
    task_names = list(dataloaders.keys())
    max_batches = max(len(loader) for loader in dataloaders.values())

    for _ in range(max_batches * len(task_names)):
        task = random.choice(task_names)
        try:
            batch = next(task_iters[task])
        except StopIteration:
            task_iters[task] = iter(dataloaders[task])
            batch = next(task_iters[task])

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, task=task)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        (task_weights[task] * loss).backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


def run_ablation_study():
    # ✅ SET BEST HYPERPARAMETERS HERE
    best_lr = 2e-5
    best_dropout = 0.1
    epoch_count = 5
    main_batch_size = 32
    aux_batch_size = 8

    aux_configs = [
        {"stereo": True, "sarcasm": True, "fine": True},
        {"stereo": False, "sarcasm": False, "fine": True},
        {"stereo": True, "sarcasm": True, "fine": False},
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    final_results = []

    for config_id, aux_config in enumerate(aux_configs):
        print(f"\n🚀 Starting config #{config_id + 1}: {aux_config}")

        aw_stereo = 1.0 if aux_config["stereo"] else 0.0
        aw_sarcasm = 1.0 if aux_config["sarcasm"] else 0.0
        aw_fine = 1.0 if aux_config["fine"] else 0.0

        # Prepare model
        model = MultiTaskBERT(dropout=best_dropout).to(device)
        optimizer = AdamW(model.parameters(), lr=best_lr)

        # === Load data (train + val) ===
        main_all = LatentHatredDataset("datasets/latent_hatred_3class_train.csv")
        main_test = LatentHatredDataset("datasets/latent_hatred_3class_test.csv")

        train_loaders = {
            "main": DataLoader(BaseTextDataset(main_all.texts, main_all.labels), batch_size=main_batch_size, shuffle=True)
        }
        test_loaders = {
            "main": DataLoader(BaseTextDataset(main_test.texts, main_test.labels), batch_size=main_batch_size)
        }

        if aw_stereo > 0:
            stereo_all = StereoSetDataset("datasets/stereoset_train.csv")
            stereo_test = StereoSetDataset("datasets/stereoset_test.csv")
            train_loaders["stereo"] = DataLoader(BaseTextDataset(stereo_all.texts, stereo_all.labels), batch_size=aux_batch_size, shuffle=True)
            test_loaders["stereo"] = DataLoader(BaseTextDataset(stereo_test.texts, stereo_test.labels), batch_size=aux_batch_size)

        if aw_sarcasm > 0:
            sarcasm_all = ISarcasmDataset("datasets/isarcasm_train.csv")
            sarcasm_test = ISarcasmDataset("datasets/isarcasm_test.csv")
            train_loaders["sarcasm"] = DataLoader(BaseTextDataset(sarcasm_all.texts, sarcasm_all.labels), batch_size=aux_batch_size, shuffle=True)
            test_loaders["sarcasm"] = DataLoader(BaseTextDataset(sarcasm_test.texts, sarcasm_test.labels), batch_size=aux_batch_size)

        if aw_fine > 0:
            fine_all = ImplicitFineHateDataset("datasets/implicit_fine_labels_train.csv")
            fine_test = ImplicitFineHateDataset("datasets/implicit_fine_labels_test.csv")
            train_loaders["implicit_fine"] = DataLoader(BaseTextDataset(fine_all.texts, fine_all.labels), batch_size=aux_batch_size, shuffle=True)
            test_loaders["implicit_fine"] = DataLoader(BaseTextDataset(fine_test.texts, fine_test.labels), batch_size=aux_batch_size)

        task_weights = {
            "main": 1.0,
            "stereo": aw_stereo,
            "sarcasm": aw_sarcasm,
            "implicit_fine": aw_fine
        }

        # === Training ===
        for epoch in range(1, epoch_count + 1):
            print(f"\n🔁 Epoch {epoch}/{epoch_count} for config #{config_id + 1}")
            loss = train_one_epoch(model, train_loaders, optimizer, task_weights, device)
            print(f"Loss: {loss:.4f}")

        # === Final Evaluation on Test Set ===
        test_metrics = evaluate(model, test_loaders, device)
        print("\n📈 Final Test Metrics:")
        for task, m in test_metrics.items():
            print(f"{task}: Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}, Precision={m['precision']:.4f}, Recall={m['recall']:.4f}")

        # === Save Model & Results ===
        config_tag = f"cfg{config_id+1}_lr{best_lr}_do{best_dropout}"
        model_filename = f"ablation_model_{config_tag}.pt"
        save_checkpoint(model, optimizer, epoch_count, model_filename)
        print(f"💾 Saved model to {model_filename}")

        for task, scores in test_metrics.items():
            final_results.append({
                "config_id": config_id + 1,
                "lr": best_lr,
                "dropout": best_dropout,
                "task": task,
                **scores
            })

    # Save test metrics
    df = pd.DataFrame(final_results)
    df.to_csv("ablation_test_metrics.csv", index=False)
    print("\n✅ Ablation test results saved to ablation_test_metrics.csv")


if __name__ == "__main__":
    run_ablation_study()