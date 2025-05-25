# ~/codes-v2/scripts/utils.py (REVISED for Multi-Task, closer to Single-Task Utils)
import re
from transformers import BertTokenizer
import torch
import os
# import glob # Not used if not listing files
import json
# import shutil # Not used if not moving/deleting files
from datetime import datetime
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report, confusion_matrix)

# --- Text Preprocessing and Encoding (from your single-task utils) ---
# Assuming a global tokenizer instance for now, like in your single-task utils.
# If your dataset loaders instantiate their own, this global one might not be used by them.
# For consistency with your provided single-task utils.py:
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

def encode_text(text, max_len=128): # Uses global tokenizer_for_utils
    """Enhanced text encoding with error handling, using global tokenizer"""
    try:
        return tokenizer_for_utils( # Uses the global one
            text,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )
    except Exception as e:
        print(f"Warning: Error encoding text '{str(text)[:50]}...': {e}")
        return tokenizer_for_utils( # Fallback with global tokenizer
            "",
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt"
        )

# --- CheckpointManager (Closely based on your Single-Task CheckpointManager) ---
class CheckpointManager:
    """
    Checkpoint management system adapted for multi-task trials.
    Manages checkpoints within a specific experiment_dir for a single trial.
    """
    def __init__(self, trial_experiment_dir, # Renamed from base_dir for clarity in this context
                       trial_name_for_logging=None): # Renamed from experiment_name
        """
        Args:
            trial_experiment_dir: The specific directory for this trial's checkpoints and artifacts.
                                  This is GIVEN by the hyperparameter searcher.
            trial_name_for_logging: A name for logging purposes, not for directory creation.
        """
        self.experiment_dir = Path(trial_experiment_dir) # This IS the specific trial's directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # These will store info about the best state *within this trial*
        self.best_metrics_for_trial = {} # Stores the metrics dict of the best epoch for this trial
        self.best_checkpoint_path_for_trial = None

        log_name = trial_name_for_logging if trial_name_for_logging else self.experiment_dir.name
        print(f"📁 Checkpoint manager initialized for trial directory: {self.experiment_dir} (Log name: {log_name})")

    def save_checkpoint(self, model, optimizer, epoch,
                        current_epoch_metrics_all_tasks=None, # Dict: {'main':{...}, 'stereo':{...}}
                        config_args=None, # argparse.Namespace or dict
                        is_best_for_this_trial=False): # Determined by train.py's logic
        """
        Save checkpoint.
        Args:
            current_epoch_metrics_all_tasks: Metrics for all tasks for the current epoch.
            is_best_for_this_trial: Boolean, true if this epoch is the best *for this specific trial*.
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

        # Save epoch-specific checkpoint (mimicking your single-task behavior)
        # For multi-task, this might be less crucial if HPO focuses on best/last.
        # But keeping for similarity.
        epoch_path = self.experiment_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint_data, epoch_path)
        # print(f"  Saved epoch-specific checkpoint: {epoch_path.name}")

        # Save best checkpoint FOR THIS TRIAL
        if is_best_for_this_trial and current_epoch_metrics_all_tasks:
            best_path = self.experiment_dir / "checkpoint_best_model.pt" # Standard name for best in trial
            torch.save(checkpoint_data, best_path)
            self.best_checkpoint_path_for_trial = best_path
            self.best_metrics_for_trial = current_epoch_metrics_all_tasks.copy() # Store metrics of this best epoch
            
            # The primary task's key metric value (that made it "best") should be clear from train.py's log
            print(f"🏆 Best checkpoint for this trial saved: {best_path.name} (Epoch {epoch})")

            # Save best metrics summary separately for easy access by HPO (as in your single task)
            # This file becomes crucial for the HPO to know the "best outcome of this trial"
            best_metrics_summary_path = self.experiment_dir / "best_metrics_summary_for_trial.json"
            with open(best_metrics_summary_path, 'w') as f:
                json.dump({
                    "best_epoch_for_this_trial": epoch,
                    "metrics_all_tasks_at_best_epoch": self.best_metrics_for_trial,
                    "config_args": config_dict, # Config for this trial
                    "timestamp": timestamp
                }, f, indent=2)
        
        # Also save a "last epoch" checkpoint for easier resuming of a trial
        last_epoch_path = self.experiment_dir / "checkpoint_last_epoch.pt"
        torch.save(checkpoint_data, last_epoch_path)
        # print(f"  Saved last epoch checkpoint: {last_epoch_path.name}")


        return epoch_path # Or best_path, or last_epoch_path depending on what's most relevant

    def load_checkpoint(self, model, optimizer=None, checkpoint_type="best", device='cpu'):
        """
        Load checkpoint. 'checkpoint_type' can be "best", "last_epoch", or an epoch number.
        Args:
            checkpoint_type: "best" (for checkpoint_best_model.pt),
                             "last_epoch" (for checkpoint_last_epoch.pt),
                             or specific epoch number (int for checkpoint_epoch_{N}.pt).
            device: Device to map storage to.
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
                 # If it's a relative path, assume it's within experiment_dir
                 checkpoint_path = self.experiment_dir / checkpoint_type


        if not checkpoint_path.exists():
            print(f"⚠️ Checkpoint file not found: {checkpoint_path}. Cannot load.")
            return None # Return None to indicate failure

        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

            if optimizer and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            epoch = checkpoint.get("epoch", 0)
            metrics_loaded = checkpoint.get("metrics_all_tasks_at_epoch", {})
            # config_loaded = checkpoint.get("config_args", {})

            print(f"✅ Loaded checkpoint: {checkpoint_path.name} (from epoch {epoch})")
            # if metrics_loaded.get("main"): # Example: print main task F1 if available
            #     main_f1 = metrics_loaded["main"].get("f1", "N/A")
            #     print(f"   Metrics at save (e.g., Main F1): {main_f1 if main_f1 == 'N/A' else f'{main_f1:.4f}'}")
            return checkpoint # Return the full dict

        except Exception as e:
            print(f"❌ Failed to load checkpoint {checkpoint_path}: {e}")
            # raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}") # Or return None
            return None


# --- Metrics Computation (Closely based on your single-task version) ---
def compute_task_metrics_detailed(y_true, y_pred, class_names=None, average_method='macro', task_name=""):
    """
    Compute comprehensive classification metrics for a single task.
    'average_method' applies to precision, recall, F1 overall scores.
    """
    # Handle empty inputs (same as before)
    if not y_true or not y_pred or len(y_true) == 0:
        print(f"Warning: Empty y_true or y_pred for task '{task_name}'. Returning zero metrics.")
        num_classes = len(class_names) if class_names else 0
        zero_per_class = [0.0] * num_classes
        return {
            'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'loss': None,
            'precision_per_class': zero_per_class, 'recall_per_class': zero_per_class,
            'f1_per_class': zero_per_class,
            'confusion_matrix': [[0]*num_classes for _ in range(num_classes)] if num_classes > 0 else [],
            'classification_report_dict': {},
            'class_names_for_report': class_names or [], # Store the original class names provided
            'average_method': average_method
        }

    accuracy = accuracy_score(y_true, y_pred)
    # Overall metrics
    precision = precision_score(y_true, y_pred, average=average_method, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average_method, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average_method, zero_division=0)

    # Determine labels present in the data for per-class metrics and report
    # This ensures sklearn functions don't fail if some classes have no samples.
    labels_present_in_data = sorted(list(set(y_true) | set(y_pred)))

    if not class_names: # If no class_names provided, generate defaults based on data
        if labels_present_in_data:
            class_names_for_report = [f"Class_{i}" for i in range(max(labels_present_in_data) + 1)]
            # Ensure labels_for_sklearn_report are within the bounds of generated class_names_for_report
            labels_for_sklearn_report = [l for l in labels_present_in_data if l < len(class_names_for_report)]
            target_names_for_sklearn_report = [class_names_for_report[l] for l in labels_for_sklearn_report]
        else: # No data, no class_names
            class_names_for_report = []
            labels_for_sklearn_report = []
            target_names_for_sklearn_report = []
    else: # class_names were provided
        class_names_for_report = class_names
        # Filter labels_present_in_data to be within the bounds of provided class_names
        labels_for_sklearn_report = [l for l in labels_present_in_data if l < len(class_names_for_report)]
        target_names_for_sklearn_report = [class_names_for_report[l] for l in labels_for_sklearn_report]


    # Per-class metrics (only for labels present in data and covered by class_names)
    if labels_for_sklearn_report: # Only compute if there are valid labels to report on
        precision_pc = precision_score(y_true, y_pred, labels=labels_for_sklearn_report, average=None, zero_division=0)
        recall_pc = recall_score(y_true, y_pred, labels=labels_for_sklearn_report, average=None, zero_division=0)
        f1_pc = f1_score(y_true, y_pred, labels=labels_for_sklearn_report, average=None, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=labels_for_sklearn_report)
        try:
            report_dict = classification_report(y_true, y_pred, labels=labels_for_sklearn_report,
                                             target_names=target_names_for_sklearn_report,
                                             output_dict=True, zero_division=0)
        except Exception as e_rep: # Catch potential errors in classification_report itself
            print(f"Warning: Could not generate classification_report for task '{task_name}': {e_rep}")
            report_dict = {}
    else: # No valid labels to compute per-class metrics for
        num_eff_classes = len(class_names_for_report)
        precision_pc = [0.0] * num_eff_classes
        recall_pc = [0.0] * num_eff_classes
        f1_pc = [0.0] * num_eff_classes
        cm = [[0] * num_eff_classes for _ in range(num_eff_classes)] if num_eff_classes > 0 else []
        report_dict = {}


    metrics = {
        'accuracy': accuracy,
        'precision': precision, # This is the 'average_method' one
        'recall': recall,
        'f1': f1,
        'loss': None, # To be filled by the evaluate function
        # The per_class arrays correspond to 'labels_for_sklearn_report'
        'precision_per_class': precision_pc.tolist() if hasattr(precision_pc, 'tolist') else precision_pc,
        'recall_per_class': recall_pc.tolist() if hasattr(recall_pc, 'tolist') else recall_pc,
        'f1_per_class': f1_pc.tolist() if hasattr(f1_pc, 'tolist') else f1_pc,
        'confusion_matrix': cm.tolist() if hasattr(cm, 'tolist') else cm, # Corresponds to 'labels_for_sklearn_report'
        'classification_report_dict': report_dict, # Keys are target_names_for_sklearn_report
        'class_names_for_report': class_names_for_report, # The full list of class names (original or generated)
        'labels_reported_per_class': labels_for_sklearn_report, # Actual label indices for which per-class was computed
        'average_method': average_method
    }
    return metrics

# --- Metrics Printing (Closely based on your single-task version) ---
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

    # Per-class metrics using classification_report_dict
    report_dict = metrics_for_one_task.get('classification_report_dict', {})
    # The keys in report_dict are the actual class names used in the report
    # (e.g., "not_hate", "implicit_hate" if target_names were provided and valid)
    
    # Get the class names that were actually reported on.
    # These are the keys in report_dict that are dictionaries themselves (not 'accuracy', 'macro avg', etc.)
    reported_class_metric_keys = [k for k, v in report_dict.items() if isinstance(v, dict)]

    if reported_class_metric_keys:
        print(f"\n  Per-Class Performance Details:")
        for class_key in reported_class_metric_keys:
            class_metrics = report_dict[class_key]
            print(f"    {class_key}:") # class_key is the actual name like "not_hate"
            print(f"      Precision: {class_metrics.get('precision', 0.0):.4f}")
            print(f"      Recall:    {class_metrics.get('recall', 0.0):.4f}")
            print(f"      F1-Score:  {class_metrics.get('f1-score', 0.0):.4f}")
            print(f"      Support:   {class_metrics.get('support', 0)}")
    
    # Optionally print confusion matrix if needed, using 'labels_reported_per_class' for row/col headers
    # This part was in your original single-task utils.py
    cm = metrics_for_one_task.get('confusion_matrix')
    labels_for_cm_headers = metrics_for_one_task.get('labels_reported_per_class', [])
    all_class_names = metrics_for_one_task.get('class_names_for_report', [])
    
    if cm and labels_for_cm_headers and all_class_names:
        # Get the names for the labels actually in the CM
        cm_header_names = [all_class_names[l] for l in labels_for_cm_headers if l < len(all_class_names)]
        if cm_header_names: # Ensure we have names for the CM headers
            print(f"\n  Confusion Matrix (for reported labels):")
            # Print header row
            header_str = "        " + "  ".join([f"{name[:7]:>7}" for name in cm_header_names]) # Truncate names if too long
            print(header_str)
            for i, row_label_idx in enumerate(labels_for_cm_headers):
                row_name = all_class_names[row_label_idx] if row_label_idx < len(all_class_names) else f"L{row_label_idx}"
                row_str = f"{row_name[:7]:>7} " # Truncate name
                if i < len(cm): # Ensure row index is valid for cm
                    row_str += " ".join([f"{val:>8}" for val in cm[i]])
                    print(row_str)
    print(underline)


# --- JSON Saving Utility (General purpose) ---
def save_json_to_file(data_dict, filepath):
    """Save a dictionary to a JSON file, attempting to make it serializable."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    def make_serializable(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        # Add other non-serializable types if needed
        return obj

    serializable_data = make_serializable(data_dict)
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    # print(f"💾 JSON data saved to: {filepath}") # train.py can print this