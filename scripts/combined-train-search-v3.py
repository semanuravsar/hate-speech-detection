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

def run_grid_search():
    learning_rates = [2e-5]
    dropouts = [0.1]
    epoch_count = 5
    main_weights = [1.0]
    aux_weights = [0.5, 1.0, 2.0]
    main_batch_size = 32
    aux_batch_size = 8

    aux_configs = [
        {"stereo": True, "sarcasm": True, "fine": True},
        {"stereo": False, "sarcasm": False, "fine": True},
        {"stereo": True, "sarcasm": True, "fine": False},
        {"stereo": False, "sarcasm": False, "fine": False},
    ]

    results = []
    best_overall_f1 = 0
    best_model_state = None
    best_model_info = ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for lr, dropout, mw, aw, aux_config in itertools.product(
        learning_rates, dropouts, main_weights, aux_weights, aux_configs
    ):
        start_time = time.time()

        aw_stereo = aw if aux_config["stereo"] else 0
        aw_sarcasm = aw if aux_config["sarcasm"] else 0
        aw_fine = aw if aux_config["fine"] else 0

        print(f"\n🚀 Starting new config:")
        print(f"Learning rate: {lr}, Dropout: {dropout}, Main batch size: {main_batch_size}, Aux batch size: {aux_batch_size}")
        print(f"Main weight: {mw}, Stereo weight: {aw_stereo}, Sarcasm weight: {aw_sarcasm}, Fine-grained weight: {aw_fine}")

        model = MultiTaskBERT(dropout=dropout).to(device)
        optimizer = AdamW(model.parameters(), lr=lr)

        main_all = LatentHatredDataset("/home/avsar/codes-v2/datasets/latent_hatred_3class_train.csv")
        main_train_texts, main_val_texts, main_train_labels, main_val_labels = train_test_split(
            main_all.texts, main_all.labels, test_size=0.2, stratify=main_all.labels, random_state=42
        )
        main_test = LatentHatredDataset("/home/avsar/codes-v2/datasets/latent_hatred_3class_test.csv")

        print(f"📊 Dataset sizes for MAIN task:")
        print(f"Train: {len(main_train_texts)} samples")
        print(f"Val:   {len(main_val_texts)} samples")
        print(f"Test:  {len(main_test)} samples")

        test_loaders = {
            "main": DataLoader(BaseTextDataset(main_test.texts, main_test.labels), batch_size=main_batch_size)
        }

        train_loaders = {
            "main": DataLoader(BaseTextDataset(main_train_texts, main_train_labels), batch_size=main_batch_size, shuffle=True)
        }
        val_loaders = {
            "main": DataLoader(BaseTextDataset(main_val_texts, main_val_labels), batch_size=main_batch_size)
        }

        if aw_stereo > 0:
            stereo_all = StereoSetDataset("/home/avsar/codes-v2/datasets/stereoset_train.csv")
            tr_texts, val_texts, tr_labels, val_labels = train_test_split(
                stereo_all.texts, stereo_all.labels, test_size=0.2, stratify=stereo_all.labels, random_state=42)
            test_stereo = StereoSetDataset("/home/avsar/codes-v2/datasets/stereoset_test.csv")
            train_loaders["stereo"] = DataLoader(BaseTextDataset(tr_texts, tr_labels), batch_size=aux_batch_size, shuffle=True)
            val_loaders["stereo"] = DataLoader(BaseTextDataset(val_texts, val_labels), batch_size=aux_batch_size)
            test_loaders["stereo"] = DataLoader(BaseTextDataset(test_stereo.texts, test_stereo.labels), batch_size=aux_batch_size)
        if aw_sarcasm > 0:
            sarcasm_all = ISarcasmDataset("/home/avsar/codes-v2/datasets/isarcasm_train.csv")
            tr_texts, val_texts, tr_labels, val_labels = train_test_split(
                sarcasm_all.texts, sarcasm_all.labels, test_size=0.2, stratify=sarcasm_all.labels, random_state=42)
            test_sarcasm = ISarcasmDataset("/home/avsar/codes-v2/datasets/isarcasm_test.csv")
            train_loaders["sarcasm"] = DataLoader(BaseTextDataset(tr_texts, tr_labels), batch_size=aux_batch_size, shuffle=True)
            val_loaders["sarcasm"] = DataLoader(BaseTextDataset(val_texts, val_labels), batch_size=aux_batch_size)
            test_loaders["sarcasm"] = DataLoader(BaseTextDataset(test_sarcasm.texts, test_sarcasm.labels), batch_size=aux_batch_size)
        if aw_fine > 0:
            fine_all = ImplicitFineHateDataset("/home/avsar/codes-v2/datasets/implicit_fine_labels_train.csv")
            tr_texts, val_texts, tr_labels, val_labels = train_test_split(
                fine_all.texts, fine_all.labels, test_size=0.2, stratify=fine_all.labels, random_state=42)
            test_fine = ImplicitFineHateDataset("/home/avsar/codes-v2/datasets/implicit_fine_labels_test.csv")
            train_loaders["implicit_fine"] = DataLoader(BaseTextDataset(tr_texts, tr_labels), batch_size=aux_batch_size, shuffle=True)
            val_loaders["implicit_fine"] = DataLoader(BaseTextDataset(val_texts, val_labels), batch_size=aux_batch_size)
            test_loaders["implicit_fine"] = DataLoader(BaseTextDataset(test_fine.texts, test_fine.labels), batch_size=aux_batch_size)

        task_weights = {"main": mw}
        if "stereo" in train_loaders: task_weights["stereo"] = aw_stereo
        if "sarcasm" in train_loaders: task_weights["sarcasm"] = aw_sarcasm
        if "implicit_fine" in train_loaders: task_weights["implicit_fine"] = aw_fine

        for epoch in range(1, epoch_count + 1):
            print(f"\n🔁 Epoch {epoch}/{epoch_count} | config: stereo={aw_stereo}, sarcasm={aw_sarcasm}, fine={aw_fine}")
            loss = train_one_epoch(model, train_loaders, optimizer, task_weights, device)
            print(f"Loss: {loss:.4f}")

            metrics = evaluate(model, val_loaders, device)
            test_metrics = evaluate(model, test_loaders, device)
            main_f1 = metrics["main"]["f1"]
            if main_f1 > best_overall_f1:
                best_overall_f1 = main_f1
                best_model_state = model.state_dict()
                best_model_info = f"lr{lr}_do{dropout}_mb{main_batch_size}_ab{aux_batch_size}_mw{mw}_st{aw_stereo}_sa{aw_sarcasm}_fi{aw_fine}"
                print(f"✅ New best overall model at epoch {epoch} with main F1 = {main_f1:.4f}")

            print("📊 Validation Metrics:")
            for task, m in metrics.items():
                print(f"{task}: Acc = {m['accuracy']:.4f}, F1 = {m['f1']:.4f}, Precision = {m['precision']:.4f}, Recall = {m['recall']:.4f}")

            print("📈 Test Metrics:")
            for task, m in test_metrics.items():
                print(f"{task}: Acc = {m['accuracy']:.4f}, F1 = {m['f1']:.4f}, Precision = {m['precision']:.4f}, Recall = {m['recall']:.4f}")

            result_row = {
                "epoch": epoch, "lr": lr, "dropout": dropout,
                "main_batch_size": main_batch_size, "aux_batch_size": aux_batch_size,
                "main_weight": mw, "stereo_weight": aw_stereo, "sarcasm_weight": aw_sarcasm,
                "implicit_fine_weight": aw_fine, "loss": loss
            }
            for task, scores in metrics.items():
                for metric_name, score in scores.items():
                    result_row[f"{task}_val_{metric_name}"] = score
            for task, scores in test_metrics.items():
                for metric_name, score in scores.items():
                    result_row[f"{task}_test_{metric_name}"] = score

            results.append(result_row)

        elapsed = time.time() - start_time
        print(f"⏱️ Total time for config: {elapsed:.2f} seconds")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        save_checkpoint(model, optimizer, epoch_count, f"best_model_overall_{best_model_info}.pt")
        print(f"\n💾 Saved overall best model to best_model_overall_{best_model_info}.pt with F1 = {best_overall_f1:.4f}")

    df = pd.DataFrame(results)
    df.to_csv("grid_search_epochwise_results.csv", index=False)
    print("\n✅ All experiments completed. Results saved to grid_search_epochwise_results.csv")

if __name__ == "__main__":
    run_grid_search()