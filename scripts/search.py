import argparse
import itertools
import time
import pandas as pd
import torch

import sys
import os
sys.path.append(os.path.expanduser("~/codes-v2"))

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from models.multitask_bert import MultiTaskBERT
from scripts.dataset_loaders import LatentHatredDataset, StereoSetDataset, ISarcasmDataset
from scripts.dataset_loaders import ImplicitFineHateDataset
from scripts.utils import load_checkpoint


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


def run_experiments():
    from scripts.train import main  # avoid circular import

    learning_rates   = [3e-5]
    dropouts         = [0.1]
    batch_sizes      = [32]
    epochs_list      = [5]
    main_weights     = [1.0]
    weights   = [1.0]

    best_score = 0
    best_config = None
    results = []

    for lr, dropout, batch_size, epochs, main_w, aux_w in itertools.product(
        learning_rates, dropouts, batch_sizes, epochs_list, main_weights, weights
    ):
        print(f"\n🚀 Running: lr={lr}, dropout={dropout}, bs={batch_size}, ep={epochs}, mw={main_w}, axw={aux_w}")
        start_time = time.time()

        args = argparse.Namespace(
            dataset_dir="/home/avsar/codes-v2/datasets",
            checkpoint_path=f"checkpoint_lr{lr}_do{dropout}_bs{batch_size}_ep{epochs}_mw{main_w}_axw{aux_w}.pt",
            resume=False,
            batch_size=batch_size,
            epochs=epochs,
            main_weight=main_w,
            stereo_weight=aux_w,
            sarcasm_weight=aux_w,
            implicit_fine_weight=aux_w,
            lr=lr,
            dropout=dropout
        )

        try:
            main(args)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MultiTaskBERT(dropout=dropout).to(device)
            load_checkpoint(model, path=args.checkpoint_path)

            hate_val = LatentHatredDataset(f"{args.dataset_dir}/latent_hatred_3class_train.csv", split="val")
            stereo_val = StereoSetDataset(f"{args.dataset_dir}/stereoset_train.csv", split="val")
            sarcasm_val = ISarcasmDataset(f"{args.dataset_dir}/isarcasm_train.csv", split="val")
            implicit_fine_val = ImplicitFineHateDataset(f"{args.dataset_dir}/implicit_fine_labels_train.csv", split="val")

            dataloaders_val = {
                "main": DataLoader(hate_val, batch_size=args.batch_size),
                "stereo": DataLoader(stereo_val, batch_size=args.batch_size),
                "sarcasm": DataLoader(sarcasm_val, batch_size=args.batch_size),
                "implicit_fine": DataLoader(implicit_fine_val, batch_size=args.batch_size)
            }

            metrics = evaluate(model, dataloaders_val, device=device)
            avg_f1 = sum([m["f1"] for m in metrics.values()]) / len(metrics)

            print("📊 Validation F1 Scores:")
            for task, m in metrics.items():
                print(f"{task}: F1 = {m['f1']:.4f}, Acc = {m['accuracy']:.4f}")

            results.append({
                "lr": lr,
                "dropout": dropout,
                "batch_size": batch_size,
                "epochs": epochs,
                "main_weight": main_w,
                "stereo_weight": aux_w,
                "sarcasm_weight": aux_w,
                "implicit_fine_weight": aux_w,
                "main_f1": metrics["main"]["f1"],
                "stereo_f1": metrics["stereo"]["f1"],
                "sarcasm_f1": metrics["sarcasm"]["f1"],
                "implicit_fine_f1": metrics["implicit_fine"]["f1"],
                "avg_f1": avg_f1
            })

            if  metrics["main"]["f1"] > best_score:
                best_score =  metrics["main"]["f1"]
                best_config = {
                    "lr": lr, "dropout": dropout, "batch_size": batch_size,
                    "epochs": epochs, "main_weight": main_w,
                    "stereo_weight": aux_w, "sarcasm_weight": aux_w,
                    "implicit_fine_weight": aux_w
                }

            print(f"✅ Finished in {time.time() - start_time:.1f}s")

        except Exception as e:
            print(f"❌ Skipped config due to error: {e}")
            continue

    df = pd.DataFrame(results)
    df.to_csv("search_results.csv", index=False)
    print(f"\n✅ Best main F1: {best_score:.4f} with config:")
    print(best_config)
    print("🔍 Full results saved to search_results.csv")


if __name__ == "__main__":
    run_experiments()