import os, sys, time, argparse, itertools, random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.append(os.path.expanduser("~/codes-v2"))

from models.multitask_bert import MultiTaskBERT
from scripts.dataset_loaders import (
    LatentHatredDataset, StereoSetDataset, ISarcasmDataset, ImplicitFineHateDataset
)
from scripts.utils import save_checkpoint, load_checkpoint

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

def run_grid_search():
    learning_rates = [2e-5]
    dropouts = [0.1]
    batch_sizes = [8]
    epoch_count = 2
    main_weights = [1.0]
    aux_weights = [1.0]

    aux_configs = [
        {"stereo": True, "sarcasm": True, "fine": True},
        {"stereo": False, "sarcasm": False, "fine": True},
        {"stereo": True, "sarcasm": True, "fine": False},
        {"stereo": False, "sarcasm": False, "fine": False},
    ]

    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for lr, dropout, batch_size, mw, aw, aux_config in itertools.product(
        learning_rates, dropouts, batch_sizes, main_weights, aux_weights, aux_configs
    ):
        start_time = time.time()

        aw_stereo = aw if aux_config["stereo"] else 0
        aw_sarcasm = aw if aux_config["sarcasm"] else 0
        aw_fine = aw if aux_config["fine"] else 0

        print(f"\n🚀 Starting new config:")
        print(f"Learning rate: {lr}, Dropout: {dropout}, Batch size: {batch_size}")
        print(f"Main weight: {mw}, Stereo weight: {aw_stereo}, Sarcasm weight: {aw_sarcasm}, Fine-grained weight: {aw_fine}")

        model = MultiTaskBERT(dropout=dropout).to(device)
        optimizer = AdamW(model.parameters(), lr=lr)

        train_loaders = {
            "main": DataLoader(LatentHatredDataset("/home/avsar/codes-v2/datasets/latent_hatred_3class_train.csv", split="train"), batch_size=batch_size, shuffle=True)
        }
        val_loaders = {
            "main": DataLoader(LatentHatredDataset("/home/avsar/codes-v2/datasets/latent_hatred_3class_train.csv", split="val"), batch_size=batch_size)
        }

        if aw_stereo > 0:
            train_loaders["stereo"] = DataLoader(StereoSetDataset("/home/avsar/codes-v2/datasets/stereoset_train.csv", split="train"), batch_size=batch_size, shuffle=True)
            val_loaders["stereo"] = DataLoader(StereoSetDataset("/home/avsar/codes-v2/datasets/stereoset_train.csv", split="val"), batch_size=batch_size)
        if aw_sarcasm > 0:
            train_loaders["sarcasm"] = DataLoader(ISarcasmDataset("/home/avsar/codes-v2/datasets/isarcasm_train.csv", split="train"), batch_size=batch_size, shuffle=True)
            val_loaders["sarcasm"] = DataLoader(ISarcasmDataset("/home/avsar/codes-v2/datasets/isarcasm_train.csv", split="val"), batch_size=batch_size)
        if aw_fine > 0:
            train_loaders["implicit_fine"] = DataLoader(ImplicitFineHateDataset("/home/avsar/codes-v2/datasets/implicit_fine_labels_train.csv", split="train"), batch_size=batch_size, shuffle=True)
            val_loaders["implicit_fine"] = DataLoader(ImplicitFineHateDataset("/home/avsar/codes-v2/datasets/implicit_fine_labels_train.csv", split="val"), batch_size=batch_size)

        task_weights = {"main": mw}
        if "stereo" in train_loaders: task_weights["stereo"] = aw_stereo
        if "sarcasm" in train_loaders: task_weights["sarcasm"] = aw_sarcasm
        if "implicit_fine" in train_loaders: task_weights["implicit_fine"] = aw_fine

        best_f1 = 0
        best_model_path = f"best_model_lr{lr}_do{dropout}_bs{batch_size}_mw{mw}_st{aw_stereo}_sa{aw_sarcasm}_fi{aw_fine}.pt"

        for epoch in range(1, epoch_count + 1):
            print(f"\n🔁 Epoch {epoch}/{epoch_count} | config: stereo={aw_stereo}, sarcasm={aw_sarcasm}, fine={aw_fine}")
            loss = train_one_epoch(model, train_loaders, optimizer, task_weights, device)
            print(f"Loss: {loss:.4f}")

            metrics = evaluate(model, val_loaders, device)
            main_f1 = metrics["main"]["f1"]
            if main_f1 > best_f1:
                best_f1 = main_f1
                save_checkpoint(model, optimizer, epoch, best_model_path)
                print(f"✅ Saved new best model at epoch {epoch} with main F1 = {main_f1:.4f}")

            print("📊 Validation Metrics:")
            for task, m in metrics.items():
                print(f"{task}: Acc = {m['accuracy']:.4f}, F1 = {m['f1']:.4f}, Precision = {m['precision']:.4f}, Recall = {m['recall']:.4f}")

            result_row = {
                "epoch": epoch, "lr": lr, "dropout": dropout, "batch_size": batch_size,
                "main_weight": mw, "stereo_weight": aw_stereo, "sarcasm_weight": aw_sarcasm,
                "implicit_fine_weight": aw_fine, "loss": loss
            }
            for task, scores in metrics.items():
                for metric_name, score in scores.items():
                    result_row[f"{task}_{metric_name}"] = score

            results.append(result_row)

        elapsed = time.time() - start_time
        print(f"⏱️ Total time for config: {elapsed:.2f} seconds")

    df = pd.DataFrame(results)
    df.to_csv("grid_search_epochwise_results.csv", index=False)
    print("\n✅ All experiments completed. Results saved to grid_search_epochwise_results.csv")

if __name__ == "__main__":
    run_grid_search()