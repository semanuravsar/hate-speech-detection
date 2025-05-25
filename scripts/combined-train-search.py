import os, sys, time, argparse, itertools
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
    dropouts = [0.1, 0.3]
    batch_sizes = [8]
    epoch_count = 5
    main_weights = [1.0]
    aux_weights = [0.5]

    results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for lr, dropout, batch_size, mw, aw in itertools.product(
        learning_rates, dropouts, batch_sizes, main_weights, aux_weights
    ):
        print(f"\n🚀 Config: lr={lr}, dropout={dropout}, bs={batch_size}, mw={mw}, aw={aw}")
        model = MultiTaskBERT(dropout=dropout).to(device)
        optimizer = AdamW(model.parameters(), lr=lr)

        train_loaders = {
            "main": DataLoader(LatentHatredDataset("datasets/latent_hatred_3class_train.csv", split="train"), batch_size=batch_size, shuffle=True),
            "stereo": DataLoader(StereoSetDataset("datasets/stereoset_train.csv", split="train"), batch_size=batch_size, shuffle=True),
            "sarcasm": DataLoader(ISarcasmDataset("datasets/isarcasm_train.csv", split="train"), batch_size=batch_size, shuffle=True),
            "implicit_fine": DataLoader(ImplicitFineHateDataset("datasets/implicit_fine_labels_train.csv", split="train"), batch_size=batch_size, shuffle=True)
        }

        test_loaders = {
            "main": DataLoader(LatentHatredDataset("datasets/latent_hatred_3class_test.csv", split="test"), batch_size=batch_size),
            "stereo": DataLoader(StereoSetDataset("datasets/stereoset_test.csv", split="test"), batch_size=batch_size),
            "sarcasm": DataLoader(ISarcasmDataset("datasets/isarcasm_test.csv", split="test"), batch_size=batch_size),
            "implicit_fine": DataLoader(ImplicitFineHateDataset("datasets/implicit_fine_labels_test.csv", split="test"), batch_size=batch_size)
        }

        task_weights = {
            "main": mw,
            "stereo": aw,
            "sarcasm": aw,
            "implicit_fine": aw
        }

        for epoch in range(1, epoch_count + 1):
            print(f"\n🔁 Epoch {epoch}/{epoch_count}")
            loss = train_one_epoch(model, train_loaders, optimizer, task_weights, device)
            print(f"Loss: {loss:.4f}")

            checkpoint_name = f"mtl_e{epoch}_lr{lr}_do{dropout}_bs{batch_size}_mw{mw}_aw{aw}.pt"
            save_checkpoint(model, optimizer, epoch, checkpoint_name)

            metrics = evaluate(model, test_loaders, device)
            result_row = {
                "epoch": epoch, "lr": lr, "dropout": dropout, "batch_size": batch_size,
                "main_weight": mw, "aux_weight": aw, "loss": loss
            }
            for task, scores in metrics.items():
                for metric_name, score in scores.items():
                    result_row[f"{task}_{metric_name}"] = score

            results.append(result_row)

    df = pd.DataFrame(results)
    df.to_csv("grid_search_epochwise_results.csv", index=False)
    print("\n✅ All experiments completed. Results saved to grid_search_epochwise_results.csv")


if __name__ == "__main__":
    import random
    run_grid_search()
