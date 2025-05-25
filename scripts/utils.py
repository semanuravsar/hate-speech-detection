# ~/codes-v2/scripts/utils.py
import numpy as np
import re
from transformers import BertTokenizer
import torch
import os
import json
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report, confusion_matrix)

# --- Text Preprocessing and Encoding ---
tokenizer_for_utils = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess_text(text):
    """Enhanced text preprocessing with better handling"""
    text = str(text).lower()
    text = re.sub(r"http\S+", "<url>", text)
    text = re.sub(r"@\w+", "<user>", text)
    text = re.sub(r"#\w+", "<hashtag>", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII
    text = re.sub(r"\s+", " ", text).strip()
    return text

def encode_text(text, max_len=128):
    """Enhanced text encoding with error handling, using global tokenizer"""
    try:
        return tokenizer_for_utils(
            text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
    except Exception as e:
        print(f"Warning: Error encoding text '{str(text)[:50]}...': {e}")
        return tokenizer_for_utils(
            "",
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )

# --- CheckpointManager ---
class CheckpointManager:
    """
    Checkpoint management system adapted for multi-task trials.
    Manages checkpoints within a specific experiment_dir for a single trial.
    """
    def __init__(self, trial_experiment_dir, trial_name_for_logging=None):
        self.experiment_dir = Path(trial_experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.best_metrics_for_trial = {}
        self.best_checkpoint_path_for_trial = None
        log_name = trial_name_for_logging if trial_name_for_logging else self.experiment_dir.name
        print(f"📁 Checkpoint manager initialized for trial directory: {self.experiment_dir} (Log name: {log_name})")

    def save_checkpoint(self, model, optimizer, epoch,
                        current_epoch_metrics_all_tasks=None,
                        config_args=None,
                        is_best_for_this_trial=False,
                        save_every_n_epochs=None): # Parameter for periodic saving
        """
        Save checkpoint.
        - Saves 'checkpoint_last_epoch.pt' always.
        - Saves 'checkpoint_best_model.pt' if 'is_best_for_this_trial' is True.
        - Optionally saves 'checkpoint_epoch_{epoch}.pt' if 'save_every_n_epochs' is set and condition met.
        """
        timestamp = datetime.now().isoformat()
        config_dict = vars(config_args) if config_args and not isinstance(config_args, dict) else config_args or {}

        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "timestamp": timestamp,
            "metrics_all_tasks_at_epoch": current_epoch_metrics_all_tasks or {},
            "config_args": config_dict
        }

        # 1. Always save/overwrite the "last epoch" checkpoint for resuming
        last_epoch_path = self.experiment_dir / "checkpoint_last_epoch.pt"
        torch.save(checkpoint_data, last_epoch_path)
        # print(f"  Saved/Updated last epoch checkpoint: {last_epoch_path.name}") # Optional verbose log

        # 2. Save "best model" checkpoint if this epoch is the best so far for this trial
        if is_best_for_this_trial and current_epoch_metrics_all_tasks:
            best_path = self.experiment_dir / "checkpoint_best_model.pt" # Standard name for best in trial
            torch.save(checkpoint_data, best_path)
            self.best_checkpoint_path_for_trial = best_path
            self.best_metrics_for_trial = current_epoch_metrics_all_tasks.copy()
            print(f"🏆 Best checkpoint for this trial saved/updated: {best_path.name} (Epoch {epoch})")

            # Save best metrics summary separately for easy access by HPO
            best_metrics_summary_path = self.experiment_dir / "best_metrics_summary_for_trial.json"
            with open(best_metrics_summary_path, 'w') as f:
                json.dump({
                    "best_epoch_for_this_trial": epoch,
                    "metrics_all_tasks_at_best_epoch": self.best_metrics_for_trial,
                    "config_args": config_dict,
                    "timestamp": timestamp
                }, f, indent=2)
            # print(f"  Saved/Updated best metrics summary: {best_metrics_summary_path.name}") # Optional

        # 3. Optionally save an epoch-specific checkpoint periodically
        returned_epoch_path = None # Initialize for the return value
        if save_every_n_epochs and (epoch % save_every_n_epochs == 0 or epoch == 1):
            # Define epoch_path (local variable for this specific file)
            epoch_path_specific = self.experiment_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint_data, epoch_path_specific)
            returned_epoch_path = epoch_path_specific # This specific epoch's path will be returned
            # print(f"  Saved periodic epoch-specific checkpoint: {epoch_path_specific.name}") # Optional

        # The return value is not currently used by train.py.
        # Returning the path of the periodically saved epoch file if it was saved,
        # or None otherwise.
        return returned_epoch_path

    def load_checkpoint(self, model, optimizer=None, checkpoint_type="best", device='cpu'):
        """
        Load checkpoint. 'checkpoint_type' can be "best", "last_epoch", or an epoch number.
        """
        if checkpoint_type == "best":
            checkpoint_path = self.experiment_dir / "checkpoint_best_model.pt"
        elif checkpoint_type == "last_epoch":
            checkpoint_path = self.experiment_dir / "checkpoint_last_epoch.pt"
        elif isinstance(checkpoint_type, int):
            checkpoint_path = self.experiment_dir / f"checkpoint_epoch_{checkpoint_type}.pt"
        else: # Assuming checkpoint_type is a direct Path object or string path
            checkpoint_path = Path(checkpoint_type)
            if not checkpoint_path.is_absolute() and not checkpoint_path.parent == self.experiment_dir:
                 checkpoint_path = self.experiment_dir / checkpoint_type

        if not checkpoint_path.exists():
            print(f"⚠️ Checkpoint file not found: {checkpoint_path}. Cannot load.")
            return None

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

            if optimizer and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            epoch = checkpoint.get("epoch", 0)
            # metrics_loaded = checkpoint.get("metrics_all_tasks_at_epoch", {}) # For optional logging
            print(f"✅ Loaded checkpoint: {checkpoint_path.name} (from epoch {epoch})")
            return checkpoint

        except Exception as e:
            print(f"❌ Failed to load checkpoint {checkpoint_path}: {e}")
            return None

# --- Metrics Computation ---
def compute_task_metrics_detailed(y_true, y_pred, class_names=None, average_method='macro', task_name=""):
    """
    Compute comprehensive classification metrics for a single task.
    """
    if not hasattr(y_true, '__len__') or not hasattr(y_pred, '__len__') or len(y_true) == 0 or len(y_pred) == 0:
        print(f"Warning: Empty or invalid y_true/y_pred for task '{task_name}'. Returning zero metrics.")
        num_classes = len(class_names) if class_names else 0
        zero_per_class = [0.0] * num_classes if num_classes > 0 else []
        return {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'loss': None,
            'precision_per_class': zero_per_class, 'recall_per_class': zero_per_class,
            'f1_per_class': zero_per_class,
            'confusion_matrix': [[0]*num_classes for _ in range(num_classes)] if num_classes > 0 else [],
            'classification_report_dict': {},
            'class_names_for_report': class_names or [],
            'average_method': average_method,
            'labels_reported_per_class': []
        }

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average_method, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average_method, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average_method, zero_division=0)

    labels_present_in_data = sorted(list(set(y_true) | set(y_pred)))

    if not class_names:
        class_names_for_report = [f"Class_{i}" for i in range(max(labels_present_in_data) + 1)] if labels_present_in_data else []
    else:
        class_names_for_report = class_names

    # Ensure labels for sklearn report are valid indices for class_names_for_report
    labels_for_sklearn_report = [l for l in labels_present_in_data if l < len(class_names_for_report)]
    target_names_for_sklearn_report = [class_names_for_report[l] for l in labels_for_sklearn_report] if labels_for_sklearn_report else []


    if labels_for_sklearn_report:
        precision_pc = precision_score(y_true, y_pred, labels=labels_for_sklearn_report, average=None, zero_division=0)
        recall_pc = recall_score(y_true, y_pred, labels=labels_for_sklearn_report, average=None, zero_division=0)
        f1_pc = f1_score(y_true, y_pred, labels=labels_for_sklearn_report, average=None, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels_for_sklearn_report)
        try:
            report_dict = classification_report(y_true, y_pred, labels=labels_for_sklearn_report,
                                             target_names=target_names_for_sklearn_report,
                                             output_dict=True, zero_division=0)
        except Exception as e_rep:
            print(f"Warning: Could not generate classification_report for task '{task_name}': {e_rep}")
            report_dict = {}
    else:
        num_eff_classes = len(class_names_for_report) if class_names_for_report else 0
        precision_pc = [0.0] * num_eff_classes
        recall_pc = [0.0] * num_eff_classes
        f1_pc = [0.0] * num_eff_classes
        cm = [[0] * num_eff_classes for _ in range(num_eff_classes)] if num_eff_classes > 0 else []
        report_dict = {}

    metrics = {
        'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'loss': None,
        'precision_per_class': precision_pc.tolist() if hasattr(precision_pc, 'tolist') else precision_pc,
        'recall_per_class': recall_pc.tolist() if hasattr(recall_pc, 'tolist') else recall_pc,
        'f1_per_class': f1_pc.tolist() if hasattr(f1_pc, 'tolist') else f1_pc,
        'confusion_matrix': cm.tolist() if hasattr(cm, 'tolist') else cm,
        'classification_report_dict': report_dict,
        'class_names_for_report': class_names_for_report,
        'labels_reported_per_class': labels_for_sklearn_report,
        'average_method': average_method
    }
    return metrics

# --- Metrics Printing ---
def print_metrics_summary(metrics_for_one_task, title="Model Performance Summary"):
    """Print a nicely formatted summary of metrics for a single task's metrics dictionary."""
    print(f"\n📊 {title}")
    underline = "=" * (len(title) + 4 if len(title) < 50 else 54)
    print(underline)

    avg_method = metrics_for_one_task.get('average_method', 'macro')
    print(f"  Overall Performance ({avg_method} average):")
    print(f"    Accuracy:  {metrics_for_one_task.get('accuracy', 0.0):.4f}")
    print(f"    Precision: {metrics_for_one_task.get('precision', 0.0):.4f}")
    print(f"    Recall:    {metrics_for_one_task.get('recall', 0.0):.4f}")
    print(f"    F1 Score:  {metrics_for_one_task.get('f1', 0.0):.4f}")
    if metrics_for_one_task.get('loss') is not None:
        print(f"    Avg Loss:  {metrics_for_one_task['loss']:.4f}")

    report_dict = metrics_for_one_task.get('classification_report_dict', {})
    reported_class_metric_keys = [k for k, v in report_dict.items() if isinstance(v, dict) and k not in ['accuracy', 'macro avg', 'weighted avg']]

    if reported_class_metric_keys:
        print(f"\n  Per-Class Performance Details:")
        for class_key in reported_class_metric_keys: # class_key is the actual name e.g., "not_hate"
            class_metrics = report_dict[class_key]
            print(f"    {class_key}:")
            print(f"      Precision: {class_metrics.get('precision', 0.0):.4f}")
            print(f"      Recall:    {class_metrics.get('recall', 0.0):.4f}")
            print(f"      F1-Score:  {class_metrics.get('f1-score', 0.0):.4f}")
            print(f"      Support:   {class_metrics.get('support', 0)}")
    
    cm = metrics_for_one_task.get('confusion_matrix')
    labels_for_cm_indices = metrics_for_one_task.get('labels_reported_per_class', []) # These are indices
    all_class_names_original = metrics_for_one_task.get('class_names_for_report', []) # Full list of names

    if cm and labels_for_cm_indices and all_class_names_original:
        # Map indices in CM to actual names for headers
        cm_header_names = [all_class_names_original[l] for l in labels_for_cm_indices if l < len(all_class_names_original)]
        
        if cm_header_names and len(cm_header_names) == len(cm): # Ensure CM dimensions match headers
            print(f"\n  Confusion Matrix (True Labels \\ Predicted Labels):")
            header_str = "Actual\\Pred" + "".join([f"{name[:7]:>8}" for name in cm_header_names])
            print(header_str)
            for i, row_label_idx in enumerate(labels_for_cm_indices):
                row_name = all_class_names_original[row_label_idx] if row_label_idx < len(all_class_names_original) else f"L{row_label_idx}"
                row_str = f"{row_name[:7]:>10}" 
                if i < len(cm):
                    row_str += "".join([f"{val:>8}" for val in cm[i]])
                    print(row_str)
    print(underline)

# --- JSON Saving Utility ---
def save_json_to_file(data_dict, filepath):
    """Save a dictionary to a JSON file, attempting to make it serializable."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def make_serializable(obj):
        if isinstance(obj, Path): return str(obj)
        if isinstance(obj, (torch.Tensor)): return obj.tolist() # Handle tensors
        if isinstance(obj, (np.ndarray)): return obj.tolist() # Handle numpy arrays
        if isinstance(obj, (np.integer)): return int(obj) # Handle numpy integers
        if isinstance(obj, (np.floating)): return float(obj) # Handle numpy floats
        if isinstance(obj, dict): return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list): return [make_serializable(i) for i in obj]
        return obj

    serializable_data = make_serializable(data_dict)
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    # print(f"💾 JSON data saved to: {filepath}")