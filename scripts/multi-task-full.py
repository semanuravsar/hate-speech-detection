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
    learning_rates = [1e-5]
    dropouts = [0.1]
    epoch_count = 5
    main_weights = [1.0]
    aux_weights = [1.0]
    main_batch_size = 32
    aux_batch_size = 8

    aux_configs = [
        {"stereo": True, "sarcasm": True, "fine": True},
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
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

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

    df = pd.DataFrame(results)
    df.to_csv("grid_search_epochwise_results.csv", index=False)
    print("\n✅ All experiments completed. Results saved to grid_search_epochwise_results.csv")


    # === 🔁 Retrain best LR per aux weight ===
    print("\n🔄 Starting retraining using best LR for each auxiliary weight...")

    final_results = []
    for aux_weight in aux_weights:
        print(f"\n🎯 Retraining for auxiliary weight = {aux_weight}")

        # Filter by aux weight and pick best LR
        subset = df[df["stereo_weight"] == aux_weight]
        if subset.empty:
            print(f"⚠️ No results found for aux_weight={aux_weight}, skipping.")
            continue
        best_row = subset.loc[subset["main_val_f1"].idxmax()]
        best_lr = best_row["lr"]
        best_dropout = best_row["dropout"]
        best_epoch = int(best_row["epoch"])  # Get from grid search results

        print(f"📌 Best LR = {best_lr}, Dropout = {best_dropout}, up to expoch = {best_epoch}")

        # Rebuild full train+val dataset
        model = MultiTaskBERT(dropout=best_dropout).to(device)
        optimizer = AdamW(model.parameters(), lr=best_lr, weight_decay=0.01)

        # Load train+val for main
        main_all = LatentHatredDataset("/home/avsar/codes-v2/datasets/latent_hatred_3class_train.csv")
        main_texts = main_all.texts
        main_labels = main_all.labels
        main_test = LatentHatredDataset("/home/avsar/codes-v2/datasets/latent_hatred_3class_test.csv")
        train_loaders = {
            "main": DataLoader(BaseTextDataset(main_texts, main_labels), batch_size=main_batch_size, shuffle=True)
        }
        test_loaders = {
            "main": DataLoader(BaseTextDataset(main_test.texts, main_test.labels), batch_size=main_batch_size)
        }

        # Load aux tasks
        if aux_weight > 0:
            stereo_all = StereoSetDataset("/home/avsar/codes-v2/datasets/stereoset_train.csv")
            stereo_test = StereoSetDataset("/home/avsar/codes-v2/datasets/stereoset_test.csv")
            sarcasm_all = ISarcasmDataset("/home/avsar/codes-v2/datasets/isarcasm_train.csv")
            sarcasm_test = ISarcasmDataset("/home/avsar/codes-v2/datasets/isarcasm_test.csv")
            fine_all = ImplicitFineHateDataset("/home/avsar/codes-v2/datasets/implicit_fine_labels_train.csv")
            fine_test = ImplicitFineHateDataset("/home/avsar/codes-v2/datasets/implicit_fine_labels_test.csv")

            train_loaders["stereo"] = DataLoader(BaseTextDataset(stereo_all.texts, stereo_all.labels), batch_size=aux_batch_size, shuffle=True)
            train_loaders["sarcasm"] = DataLoader(BaseTextDataset(sarcasm_all.texts, sarcasm_all.labels), batch_size=aux_batch_size, shuffle=True)
            train_loaders["implicit_fine"] = DataLoader(BaseTextDataset(fine_all.texts, fine_all.labels), batch_size=aux_batch_size, shuffle=True)

            test_loaders["stereo"] = DataLoader(BaseTextDataset(stereo_test.texts, stereo_test.labels), batch_size=aux_batch_size)
            test_loaders["sarcasm"] = DataLoader(BaseTextDataset(sarcasm_test.texts, sarcasm_test.labels), batch_size=aux_batch_size)
            test_loaders["implicit_fine"] = DataLoader(BaseTextDataset(fine_test.texts, fine_test.labels), batch_size=aux_batch_size)

        task_weights = {
            "main": 1.0,
            "stereo": aux_weight,
            "sarcasm": aux_weight,
            "implicit_fine": aux_weight
        }

        # Retrain
        for epoch in range(1, best_epoch + 1):
            print(f"\n🔁 Retrain Epoch {epoch}/{best_epoch}")
            loss = train_one_epoch(model, train_loaders, optimizer, task_weights, device)
            print(f"Loss: {loss:.4f}")

        # Final test evaluation
        test_metrics = evaluate(model, test_loaders, device)
        print("\n📈 Final Test Metrics:")
        for task, m in test_metrics.items():
            print(f"{task}: Acc = {m['accuracy']:.4f}, F1 = {m['f1']:.4f}, Precision = {m['precision']:.4f}, Recall = {m['recall']:.4f}")

        # Save final model
        model_filename = f"retrained_model_aux{aux_weight}_lr{best_lr}_do{best_dropout}.pt"
        save_checkpoint(model, optimizer, best_epoch, model_filename)
        print(f"💾 Saved retrained model to {model_filename}")

        # Save final test metrics for this aux_weight
        for task, scores in test_metrics.items():
            final_results.append({
                "aux_weight": aux_weight,
                "lr": best_lr,
                "dropout": best_dropout,
                "task": task,
                **scores
            })

    # Save all final results
    final_df = pd.DataFrame(final_results)
    final_df.to_csv("final_retrain_test_metrics.csv", index=False)
    print("\n✅ Final retrained test results saved to final_retrain_test_metrics.csv")


if __name__ == "__main__":
    run_grid_search()