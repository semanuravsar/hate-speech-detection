# ~/codes-v2/scripts/train.py
# (REVISED to closely mimic single-task train.py structure for a multi-task trial)

import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import time
from pathlib import Path
import json
import random # For multi-task batch sampling

import sys
import os
# Ensure project root is discoverable if running standalone
# For HPO, search_v2.py handles sys.path
# if __name__ == "__main__": # Add only if testing standalone
#     sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Assuming utils.py is in the same 'scripts' package or accessible
from .utils import CheckpointManager, compute_task_metrics_detailed, print_metrics_summary, save_json_to_file

from .dataset_loaders import LatentHatredDataset, StereoSetDataset, ImplicitFineHateDataset, ISarcasmDataset

# Adjust path if models directory is elsewhere relative to scripts
from models.multitask_bert import MultiTaskBERT

# --- Constants for this training script ---
TASK_CLASS_NAMES = {
    "main": ["not_hate", "implicit_hate", "explicit_hate"],
    "stereo": ["stereotype", "anti-stereotype", "unrelated"],
    "sarcasm": ["not_sarcasm", "sarcasm"],
    "implicit_fine": ["grievance", "incitement", "inferiority", "irony", "stereotypical", "threatening", "other"],
}
PRIMARY_VALIDATION_TASK_NAME = "main" # Task whose metric will be optimized for "best_model" in this trial
PRIMARY_METRIC_FOR_BEST_MODEL = "f1" # e.g., 'f1', 'accuracy' from the primary task

# --- train_epoch function (adapted from single-task, for multi-task) ---
def train_epoch_multitask_detailed(model, train_dataloaders, optimizer, task_weights, device,
                                   epoch_num, total_epochs, verbose_batch_logging=True):
    """Train for one epoch with progress reporting, adapted for multi-task."""
    model.train()
    total_weighted_loss_epoch = 0.0
    num_batches_processed = 0

    task_iters = {task: iter(loader) for task, loader in train_dataloaders.items()}
    task_names = list(train_dataloaders.keys())
    
    # Determine number of batches for the epoch
    # Strategy: ensure each sample from the largest dataset is seen on average once,
    # while continuously sampling from all tasks.
    if not train_dataloaders or not any(train_dataloaders.values()):
        print("Warning: No data in train_dataloaders for training epoch. Skipping.")
        return 0.0
        
    max_len_dataloader = max(len(loader) for loader in train_dataloaders.values() if loader)
    num_sampling_steps = max_len_dataloader * len(task_names) # Total batches to sample across all tasks
    
    # Fallback if calculation results in zero (e.g. all dataloaders empty)
    if num_sampling_steps == 0 and any(len(loader) > 0 for loader in train_dataloaders.values()):
        num_sampling_steps = sum(len(loader) for loader in train_dataloaders.values()) # Alternative sum of all batches
    if num_sampling_steps == 0 : # Still zero, means no data
        print("Warning: num_sampling_steps is 0. Skipping training epoch content.")
        return 0.0

    print(f"  🏋️  Training epoch {epoch_num}/{total_epochs} with ~{num_sampling_steps} mixed task batches...")

    for batch_idx in range(num_sampling_steps):
        chosen_task = random.choice(task_names)
        current_dataloader_len = len(train_dataloaders[chosen_task]) if train_dataloaders[chosen_task] else 0

        try:
            batch = next(task_iters[chosen_task])
        except StopIteration:
            task_iters[chosen_task] = iter(train_dataloaders[chosen_task])
            try:
                batch = next(task_iters[chosen_task])
            except StopIteration: # Should only occur if dataloader is empty
                continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        # CRITICAL: MultiTaskBERT.forward() must accept labels and return (loss, logits)
        loss, _ = model(input_ids, attention_mask, task=chosen_task, labels=labels)

        if loss is None:
             print(f"ERROR: Model did not return loss for task {chosen_task}. Training cannot proceed effectively.")
             # Fallback to manual calculation if possible, but ideally model handles this
             # logits_only = model(input_ids, attention_mask, task=chosen_task) # if forward can do this
             # loss = torch.nn.functional.cross_entropy(logits_only, labels)
             continue # Or raise error

        weighted_loss = task_weights[chosen_task] * loss
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient Clipping
        optimizer.step()

        total_weighted_loss_epoch += weighted_loss.item() # Use item of weighted_loss for logging
        num_batches_processed += 1

        # Progress logging (like single-task train_epoch)
        # Log more frequently if num_sampling_steps is large
        log_interval = max(1, num_sampling_steps // 20) if num_sampling_steps > 100 else 50
        if verbose_batch_logging and batch_idx > 0 and batch_idx % log_interval == 0:
            avg_loss = total_weighted_loss_epoch / num_batches_processed if num_batches_processed > 0 else 0
            # For batch progress, show relative to the current task's dataloader if meaningful,
            # or just overall batch_idx / num_sampling_steps
            print(f"    Epoch {epoch_num}, Batch {batch_idx+1}/{num_sampling_steps} - Avg Weighted Loss: {avg_loss:.4f}")

    return total_weighted_loss_epoch / num_batches_processed if num_batches_processed > 0 else 0.0


# --- evaluate_model function (adapted from single-task, for multi-task) ---
@torch.no_grad()
def evaluate_multitask_detailed(model, eval_dataloaders, device, epoch_num_for_log=None):
    """Comprehensive model evaluation for all tasks, returning detailed metrics per task."""
    model.eval()
    all_tasks_metrics_results = {} # To store detailed metrics for each task

    if epoch_num_for_log:
        print(f"  📊 Evaluating epoch {epoch_num_for_log} on validation sets...")
    else:
        print(f"  📊 Evaluating model on validation sets...")

    for task_name, dataloader in eval_dataloaders.items():
        all_preds_task, all_labels_task = [], []
        total_loss_task = 0.0
        num_batches_task = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # CRITICAL: MultiTaskBERT.forward() must accept labels and return (loss, logits)
            loss, logits = model(input_ids, attention_mask, task=task_name, labels=labels)
            preds = torch.argmax(logits, dim=1)

            all_preds_task.extend(preds.cpu().tolist())
            all_labels_task.extend(labels.cpu().tolist())
            if loss is not None:
                total_loss_task += loss.item()
            num_batches_task += 1
        
        avg_loss_task = total_loss_task / num_batches_task if num_batches_task > 0 and total_loss_task !=0 else None
        
        task_specific_class_names = TASK_CLASS_NAMES.get(task_name)
        # Use the detailed metrics computation from utils.py
        metrics_for_this_task = compute_task_metrics_detailed(
            all_labels_task, all_preds_task,
            class_names=task_specific_class_names,
            task_name=task_name # For logging within compute_task_metrics_detailed
        )
        if avg_loss_task is not None:
            metrics_for_this_task['loss'] = avg_loss_task # Add loss to the detailed metrics
        
        all_tasks_metrics_results[task_name] = metrics_for_this_task # Store the detailed dict
        
    return all_tasks_metrics_results


# --- Main training trial function (structure from single-task train_model) ---
def run_training_trial(args, trial_log_name_for_hpo=None):
    """
    Main training function for a single multi-task trial.
    Args:
        args: Training arguments (argparse.Namespace from HPO script).
        trial_log_name_for_hpo: A descriptive name for this trial from HPO script.
    """
    effective_trial_name = trial_log_name_for_hpo if trial_log_name_for_hpo else f"trial_{time.strftime('%Y%m%d_%H%M%S')}"
    print(f"🚀 Starting Multi-Task Training Trial: {effective_trial_name}")
    print(f"{'='*60}")
    print(f"Configuration (args provided):")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print(f"{'='*60}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Checkpoint Manager Setup ---
    # args.checkpoint_dir is GIVEN by the HPO script, unique for this trial.
    trial_artifacts_dir = Path(args.checkpoint_dir)
    checkpoint_manager = CheckpointManager(
        trial_experiment_dir=trial_artifacts_dir,
        trial_name_for_logging=effective_trial_name # For CheckpointManager's internal logging
    )
    # Save the exact config used for this trial (mimicking single-task)
    save_json_to_file(vars(args), trial_artifacts_dir / "trial_config_args.json")

    # --- Model and Optimizer (mimicking single-task) ---
    # Ensure MultiTaskBERT.__init__ accepts dropout_rate
    model = MultiTaskBERT(dropout_rate=args.dropout).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay # Using weight_decay from args
    )

    # --- Datasets and Dataloaders ---
    print(f"\n📊 Loading datasets from: {args.dataset_dir} ...")

    # Paths to ORIGINAL (or ORIGINAL SAMPLE) CSVs
    # Construct these based on whether HPO runs on samples or full data
    # Example: If HPO uses sample files
    hate_original_path = f"{args.dataset_dir}/latent_hatred_3class.csv"
    stereo_original_path = f"{args.dataset_dir}/stereoset.csv"
    sarcasm_original_path = f"{args.dataset_dir}/isarcasm.csv"
    fine_original_path = f"{args.dataset_dir}/implicit_fine_labels.csv" # Make sure this sample exists

    # If HPO uses full files, change paths accordingly:
    # hate_original_path = f"{args.dataset_dir}/latent_hatred_3class.csv"
    # ... etc.

    # Train datasets (Dataset classes will handle the split)
    hate_train = LatentHatredDataset(hate_original_path, split="train")
    stereo_train = StereoSetDataset(stereo_original_path, split="train")
    sarcasm_train = ISarcasmDataset(sarcasm_original_path, split="train")
    fine_train = ImplicitFineHateDataset(fine_original_path, split="train")

    # Validation datasets
    hate_val = LatentHatredDataset(hate_original_path, split="val")
    stereo_val = StereoSetDataset(stereo_original_path, split="val")
    sarcasm_val = ISarcasmDataset(sarcasm_original_path, split="val")
    fine_val = ImplicitFineHateDataset(fine_original_path, split="val")

    dataloaders_train = {
        "main": DataLoader(hate_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        "stereo": DataLoader(stereo_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        "sarcasm": DataLoader(sarcasm_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers),
        "implicit_fine": DataLoader(fine_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    }
    dataloaders_val = {
        "main": DataLoader(hate_val, batch_size=args.batch_size, num_workers=args.num_workers),
        "stereo": DataLoader(stereo_val, batch_size=args.batch_size, num_workers=args.num_workers),
        "sarcasm": DataLoader(sarcasm_val, batch_size=args.batch_size, num_workers=args.num_workers),
        "implicit_fine": DataLoader(fine_val, batch_size=args.batch_size, num_workers=args.num_workers)
    }
    print("Datasets loaded for HPO trial (with in-memory splitting).")

    task_weights = { # From args, as in original multi-task train
        "main": args.main_weight, "stereo": args.stereo_weight,
        "sarcasm": args.sarcasm_weight, "implicit_fine": args.implicit_fine_weight
    }

    # --- Resume Logic (mimicking single-task) ---
    start_epoch = 0
    best_primary_metric_val_for_this_trial = -float('inf') # For F1/Accuracy

    if args.resume: # HPO script usually sets this to False for fresh trials
        print(f"Attempting to resume trial from: {trial_artifacts_dir}")
        # Single-task used 'resume_from' type, here we'll try 'last_epoch' by default for simplicity
        loaded_checkpoint_data = checkpoint_manager.load_checkpoint(
            model, optimizer, checkpoint_type="last_epoch", device=device # From utils.py
        )
        if loaded_checkpoint_data:
            start_epoch = loaded_checkpoint_data.get("epoch", 0)
            # Restore best primary metric if available in best_metrics_summary_for_trial.json
            # This helps continue tracking the best for this specific resumed trial.
            best_summary_path = trial_artifacts_dir / "best_metrics_summary_for_trial.json"
            if best_summary_path.exists():
                try:
                    with open(best_summary_path, 'r') as f_bm:
                        bm_data = json.load(f_bm)
                    # Get metrics from the best epoch of the *previous run of this trial*
                    prior_best_metrics = bm_data.get("metrics_all_tasks_at_best_epoch", {})
                    best_primary_metric_val_for_this_trial = prior_best_metrics.get(PRIMARY_VALIDATION_TASK_NAME, {}).get(PRIMARY_METRIC_FOR_BEST_MODEL, -float('inf'))
                    print(f"  Resumed. Prior best {PRIMARY_VALIDATION_TASK_NAME} {PRIMARY_METRIC_FOR_BEST_MODEL} for this trial: {best_primary_metric_val_for_this_trial:.4f} from epoch {bm_data.get('best_epoch_for_this_trial')}")
                except Exception as e_resume_metric:
                    print(f"  Warning: Could not parse prior best metric from {best_summary_path}: {e_resume_metric}")


    print(f"\n🎯 Training trial '{effective_trial_name}' starting from epoch {start_epoch + 1} (Target total: {args.epochs} epochs)")
    
    # --- Training History Tracking (mimicking single-task) ---
    training_history_for_this_trial = []
    trial_overall_start_time = time.time()

    # --- Epoch Loop (mimicking single-task) ---
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        current_epoch_num = epoch + 1
        print(f"\n{'='*60}\nEpoch {current_epoch_num}/{args.epochs}\n{'='*60}")

        # Training phase
        avg_train_loss_epoch = train_epoch_multitask_detailed(
            model, dataloaders_train, optimizer, task_weights, device,
            epoch_num=current_epoch_num, total_epochs=args.epochs,
            verbose_batch_logging=getattr(args, 'verbose_batch_logging', True)
        )

        # Validation phase
        current_epoch_val_metrics_all_tasks = evaluate_multitask_detailed(
            model, dataloaders_val, device, epoch_num_for_log=current_epoch_num
        )

        # Log epoch results (mimicking single-task)
        epoch_duration_seconds = time.time() - epoch_start_time
        print(f"\n📈 Epoch {current_epoch_num} Results (completed in {epoch_duration_seconds:.1f}s):")
        print(f"   Train Avg Weighted Loss: {avg_train_loss_epoch:.4f}")
        # Print summary for each task using the utility
        for task_name, metrics_dict in current_epoch_val_metrics_all_tasks.items():
            # print_metrics_summary from utils.py will print detailed metrics for this task
            print_metrics_summary(metrics_dict, title=f"   Task: {task_name.upper()} - Validation Performance")
        
        # Track training history (mimicking single-task)
        history_entry = {
            'epoch': current_epoch_num,
            'avg_train_weighted_loss': avg_train_loss_epoch,
            'learning_rate': args.lr, # Or optimizer.param_groups[0]['lr']
            'epoch_time_seconds': epoch_duration_seconds,
            # Store all validation metrics for this epoch
            'validation_metrics_all_tasks': current_epoch_val_metrics_all_tasks
        }
        training_history_for_this_trial.append(history_entry)
        
        # Check if this is the best model FOR THIS TRIAL (mimicking single-task)
        current_primary_task_metrics = current_epoch_val_metrics_all_tasks.get(PRIMARY_VALIDATION_TASK_NAME, {})
        current_primary_metric_value = current_primary_task_metrics.get(PRIMARY_METRIC_FOR_BEST_MODEL, -float('inf'))
        
        is_best_epoch_for_this_trial_run = current_primary_metric_value > best_primary_metric_val_for_this_trial
        if is_best_epoch_for_this_trial_run:
            best_primary_metric_val_for_this_trial = current_primary_metric_value
            print(f"🎯 NEW BEST for this trial! {PRIMARY_VALIDATION_TASK_NAME} {PRIMARY_METRIC_FOR_BEST_MODEL}: {best_primary_metric_val_for_this_trial:.4f} at epoch {current_epoch_num}")
        
        # Save checkpoints using CheckpointManager (mimicking single-task)
        checkpoint_manager.save_checkpoint( # From utils.py
            model=model,
            optimizer=optimizer,
            epoch=current_epoch_num,
            current_epoch_metrics_all_tasks=current_epoch_val_metrics_all_tasks, # Pass all task metrics
            config_args=args, # Pass the full args
            is_best_for_this_trial=is_best_epoch_for_this_trial_run # Pass the boolean
        )
        
    # --- Training Trial Completed ---
    total_trial_duration_seconds = time.time() - trial_overall_start_time
    print(f"\n🎉 Training trial '{effective_trial_name}' completed in {total_trial_duration_seconds / 60:.1f} minutes.")
    print(f"🏆 Best {PRIMARY_VALIDATION_TASK_NAME} {PRIMARY_METRIC_FOR_BEST_MODEL} for this trial (on its val set): {best_primary_metric_val_for_this_trial:.4f}")
    
    # Save training history for this trial (mimicking single-task)
    history_path = trial_artifacts_dir / "trial_training_history.json"
    save_json_to_file(training_history_for_this_trial, history_path) # From utils.py
    
    # The single-task train_model returned a dict. For HPO, the artifacts in trial_artifacts_dir
    # (especially best_metrics_summary_for_trial.json and checkpoint_best_model.pt) are key.
    # This function doesn't strictly need to return much if HPO script reads files.
    # However, returning the key performance indicator is good for direct feedback if used standalone.
    print(f"Trial artifacts (checkpoints, history, best metrics summary) saved in: {trial_artifacts_dir}")
    # The HPO script will look for "best_metrics_summary_for_trial.json"
    # and "checkpoint_best_model.pt" in trial_artifacts_dir.

    # This main function, when called by HPO, effectively completes.
    # The HPO script will then decide what to do with the artifacts produced.

# --- HPO Entry Point (mimicking single-task main(args) -> train_model(args, experiment_name)) ---
def main(args_namespace_from_hpo, trial_log_name_from_hpo=None):
    """
    Main entry point to be called by the Hyperparameter Search Manager script.
    It runs one full training trial.
    Args:
        args_namespace_from_hpo: The argparse.Namespace object with all hyperparams and configs.
        trial_log_name_from_hpo: A descriptive name for logging and potentially for CheckpointManager.
    """
    # The `experiment_name` in single-task was used for CheckpointManager's subdir.
    # Here, `args_namespace_from_hpo.checkpoint_dir` IS that subdirectory.
    # `trial_log_name_from_hpo` is for print logs.
    run_training_trial(args_namespace_from_hpo, trial_log_name_for_hpo=trial_log_name_from_hpo)
    # No complex dictionary needs to be returned if HPO relies on file artifacts.
    # If HPO needs a direct return, structure it like single-task train_model's return.
    # For now, side effects (saved files) are the primary output for HPO.


if __name__ == "__main__":
    # This block is for testing this train.py script STANDALONE.
    # The HPO script (search_v2.py) will call the `main()` function above directly.
    parser = argparse.ArgumentParser(description="Multi-Task BERT Training Trial Script (Standalone Test)")
    
    # Required for standalone run:
    parser.add_argument("--dataset_dir", type=str, default="/home/avsar/codes-v2/datasets", help="Directory containing dataset CSVs")
    parser.add_argument("--checkpoint_dir", type=str, default="./standalone_trial_output", help="Output directory for THIS standalone trial's artifacts")
    
    # Mimic args that HPO would provide:
    parser.add_argument("--resume", action="store_true", help="Attempt to resume this standalone trial")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2) # Keep short for testing
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate for MultiTaskBERT")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for AdamW")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--verbose_batch_logging", type=bool, default=True) # For train_epoch logging

    parser.add_argument("--main_weight", type=float, default=1.0)
    parser.add_argument("--stereo_weight", type=float, default=0.3)
    parser.add_argument("--sarcasm_weight", type=float, default=0.3)
    parser.add_argument("--implicit_fine_weight", type=float, default=0.3)
    
    args_standalone = parser.parse_args()

    # Ensure the standalone output directory exists
    Path(args_standalone.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # --- Sanity check for MultiTaskBERT (from previous refinement) ---
    try:
        _test_model = MultiTaskBERT(dropout_rate=args_standalone.dropout) # Pass dropout_rate
        # Dummy inputs for a quick forward pass check
        _dummy_bs = 2
        _dummy_seq_len = 10
        _test_input_ids = torch.randint(0, _test_model.bert.config.vocab_size, (_dummy_bs, _dummy_seq_len))
        _test_attention_mask = torch.ones((_dummy_bs, _dummy_seq_len), dtype=torch.long)
        _test_labels_main = torch.randint(0, 3, (_dummy_bs,)) # Assuming 3 classes for main task
        
        _out = _test_model(_test_input_ids, _test_attention_mask, task="main", labels=_test_labels_main)
        if not (isinstance(_out, tuple) and len(_out) == 2 and isinstance(_out[0], torch.Tensor) and isinstance(_out[1], torch.Tensor)):
            print("ERROR: MultiTaskBERT forward method must return a tuple (loss, logits) when labels are provided.")
            sys.exit(1)
        print("MultiTaskBERT instantiation and forward pass (with labels) seem OK.")
    except Exception as e_model_check:
        print(f"ERROR during MultiTaskBERT sanity check: {e_model_check}")
        print("Please ensure MultiTaskBERT's __init__ accepts 'dropout_rate' and its forward(..., labels=None) returns (loss, logits) when labels are provided.")
        sys.exit(1)
    # --- End Sanity check ---

    # For standalone test, generate a simple trial log name
    standalone_trial_log_name = f"standalone_run_{Path(args_standalone.checkpoint_dir).name}"
    
    try:
        # Call the main HPO entry point for this standalone test
        main(args_standalone, trial_log_name_from_hpo=standalone_trial_log_name)
        print(f"\n✅ Standalone training trial test completed successfully!")
        print(f"   Artifacts for this trial should be in: {args_standalone.checkpoint_dir}")
    except Exception as e_standalone:
        print(f"\n❌ Standalone training trial test failed: {e_standalone}")
        import traceback
        traceback.print_exc()