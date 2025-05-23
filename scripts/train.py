import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

import sys
import os
sys.path.append(os.path.expanduser("~/a"))

from models.multitask_bert import MultiTaskBERT
from scripts.dataset_loaders import LatentHatredDataset, StereoSetDataset
from scripts.dataset_loaders import ISarcasmDataset
from scripts.utils import save_checkpoint, load_checkpoint
from sklearn.metrics import accuracy_score, f1_score

def train(model, dataloaders, optimizer, task_weights, device):
    model.train()
    total_loss = 0.0

    # Initialize iterators
    task_iters = {task: iter(loader) for task, loader in dataloaders.items()}
    active_tasks = set(dataloaders.keys())

    while active_tasks:
        for task in list(active_tasks):  # make a copy since we'll modify it
            try:
                batch = next(task_iters[task])
            except StopIteration:
                active_tasks.remove(task)
                continue  # skip to next task

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

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        metrics[task] = {"accuracy": acc, "f1": f1}
    return metrics

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskBERT().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    hate_train = LatentHatredDataset(f"{args.dataset_dir}/latent_hatred_3class_train.csv", split="train")
    hate_val   = LatentHatredDataset(f"{args.dataset_dir}/latent_hatred_3class_train.csv", split="val")

    stereo_train = StereoSetDataset(f"{args.dataset_dir}/stereoset_train.csv", split="train")
    stereo_val   = StereoSetDataset(f"{args.dataset_dir}/stereoset_train.csv", split="val")

    sarcasm_train = ISarcasmDataset(f"{args.dataset_dir}/isarcasm_train.csv", split="train")
    sarcasm_val   = ISarcasmDataset(f"{args.dataset_dir}/isarcasm_train.csv", split="val")

    dataloaders_train = {
        "main": DataLoader(hate_train, batch_size=args.batch_size, shuffle=True),
        "stereo": DataLoader(stereo_train, batch_size=args.batch_size, shuffle=True),
        "sarcasm": DataLoader(sarcasm_train, batch_size=args.batch_size, shuffle=True)
    }
    dataloaders_val = {
        "main": DataLoader(hate_val, batch_size=args.batch_size),
        "stereo": DataLoader(stereo_val, batch_size=args.batch_size),
        "sarcasm": DataLoader(sarcasm_val, batch_size=args.batch_size)
    }

    task_weights = {
        "main": args.main_weight,
        "stereo": args.stereo_weight,
        "sarcasm": args.sarcasm_weight
    }

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, args.checkpoint_path)

    print("Starting epoch loop...", flush=True)

    for epoch in range(start_epoch, args.epochs):
        loss = train(model, dataloaders_train, optimizer, task_weights, device)
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {loss:.4f}", flush=True)

        val = evaluate(model, dataloaders_val, device)
        for task, m in val.items():
            print(f"[{task}] Accuracy: {m['accuracy']:.4f}, F1: {m['f1']:.4f}", flush=True)

        save_checkpoint(model, optimizer, epoch+1, args.checkpoint_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--main_weight", type=float, default=1.0)
    parser.add_argument("--stereo_weight", type=float, default=0.2)
    parser.add_argument("--sarcasm_weight", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    args = parser.parse_args()

    main(args)
