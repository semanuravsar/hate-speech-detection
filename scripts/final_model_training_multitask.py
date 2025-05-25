# final_model_training_multitask.py
import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
import time
from pathlib import Path
import json
import random # For training epoch sampling

# Assuming 'scripts' is in PYTHONPATH or this file is run from project root
# Adjust imports based on your actual project structure
try:
    from scripts.utils import (
        # CheckpointManager, # Not strictly needed if just saving final model directly
        compute_task_metrics_detailed,
        print_metrics_summary,
        save_json_to_file
    )
    from scripts.dataset_loaders import (
        LatentHatredDataset, StereoSetDataset,
        ImplicitFineHateDataset, ISarcasmDataset
    )
    from models.multitask_bert import MultiTaskBERT
except ImportError:
    # Fallback for direct execution if scripts is a subdir
    from utils import (
        compute_task_metrics_detailed, print_metrics_summary, save_json_to_file
    )
    from dataset_loaders import (
        LatentHatredDataset, StereoSetDataset,
        ImplicitFineHateDataset, ISarcasmDataset
    )
    # Assuming models is also accessible or in PYTHONPATH
    sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # Add parent dir for 'models'
    from models.multitask_bert import MultiTaskBERT


# --- Constants (MUST match those in your scripts/train.py if reusing logic) ---
TASK_CLASS_NAMES = {
    "main": ["not_hate", "implicit_hate", "explicit_hate"],
    "stereo": ["stereotype", "anti-stereotype", "unrelated"],
    "sarcasm": ["not_sarcasm", "sarcasm"],
    "implicit_fine": ["incitement", "inferiority", "irony", "other", "stereotypical", "threatening", "white_grievance"], # Example for ImplicitFineHate
}

# --- Training Epoch function for Final Model ---
# This is copied and adapted from the 'train_epoch_multitask_detailed' in scripts/train.py
def train_final_epoch_multitask(model, train_dataloaders, optimizer, task_weights, device,
                                epoch_num, total_epochs, verbose_batch_logging=True):
    model.train()
    total_weighted_loss_epoch = 0.0
    num_batches_processed = 0

    task_iters = {task: iter(loader) for task, loader in train_dataloaders.items()}
    task_names = list(train_dataloaders.keys())

    if not train_dataloaders or not any(train_dataloaders.values()):
        print("Warning: No data in train_dataloaders for final training epoch. Skipping.")
        return 0.0
        
    max_len_dataloader = max(len(loader) for loader in train_dataloaders.values() if loader)
    num_sampling_steps = max_len_dataloader * len(task_names)
    
    if num_sampling_steps == 0 and any(len(loader) > 0 for loader in train_dataloaders.values()):
        num_sampling_steps = sum(len(loader) for loader in train_dataloaders.values())
    if num_sampling_steps == 0:
        print("Warning: num_sampling_steps is 0 for final training. Skipping epoch content.")
        return 0.0

    print(f"  🏋️  Final Training Epoch {epoch_num}/{total_epochs} with ~{num_sampling_steps} mixed task batches...")

    for batch_idx in range(num_sampling_steps):
        chosen_task = random.choice(task_names)
        try:
            batch = next(task_iters[chosen_task])
        except StopIteration:
            task_iters[chosen_task] = iter(train_dataloaders[chosen_task])
            try:
                batch = next(task_iters[chosen_task])
            except StopIteration:
                continue

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        loss, _ = model(input_ids, attention_mask, task=chosen_task, labels=labels)

        if loss is None:
             print(f"ERROR: Model did not return loss for task {chosen_task} during final training.")
             continue

        weighted_loss = task_weights[chosen_task] * loss
        weighted_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_weighted_loss_epoch += weighted_loss.item()
        num_batches_processed += 1

        log_interval = max(1, num_sampling_steps // 20) if num_sampling_steps > 100 else 50
        if verbose_batch_logging and batch_idx > 0 and batch_idx % log_interval == 0:
            avg_loss = total_weighted_loss_epoch / num_batches_processed if num_batches_processed > 0 else 0
            print(f"    Epoch {epoch_num}, Batch {batch_idx+1}/{num_sampling_steps} - Avg Weighted Loss: {avg_loss:.4f}")

    return total_weighted_loss_epoch / num_batches_processed if num_batches_processed > 0 else 0.0

# --- Evaluation function for Final Model on Test Set ---
# This is copied and adapted from 'evaluate_multitask_detailed' in scripts/train.py
@torch.no_grad()
def evaluate_final_multitask_on_test(model, test_dataloaders, device):
    model.eval()
    all_tasks_test_metrics = {}
    print(f"  📊 Evaluating final model on TEST sets...")

    for task_name, dataloader in test_dataloaders.items():
        all_preds_task, all_labels_task = [], []
        total_loss_task = 0.0
        num_batches_task = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            loss, logits = model(input_ids, attention_mask, task=task_name, labels=labels)
            preds = torch.argmax(logits, dim=1)

            all_preds_task.extend(preds.cpu().tolist())
            all_labels_task.extend(labels.cpu().tolist())
            if loss is not None:
                total_loss_task += loss.item()
            num_batches_task += 1
        
        avg_loss_task = total_loss_task / num_batches_task if num_batches_task > 0 and total_loss_task !=0 else None
        
        task_specific_class_names = TASK_CLASS_NAMES.get(task_name)
        metrics_for_this_task = compute_task_metrics_detailed( # From utils
            all_labels_task, all_preds_task,
            class_names=task_specific_class_names,
            task_name=task_name
        )
        if avg_loss_task is not None:
            metrics_for_this_task['loss'] = avg_loss_task
        
        all_tasks_test_metrics[task_name] = metrics_for_this_task
        
    return all_tasks_test_metrics


class FinalMultiTaskModelTrainer:
    def __init__(self, dataset_root_dir, best_hyperparams_dict, final_model_output_dir):
        self.dataset_root_dir = Path(dataset_root_dir)
        self.best_hyperparams = best_hyperparams_dict # This is a dictionary
        self.final_model_output_dir = Path(final_model_output_dir)
        self.final_model_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🛠️ FinalMultiTaskModelTrainer initialized.")
        print(f"  Dataset root: {self.dataset_root_dir}")
        print(f"  Best Hyperparameters to use: {json.dumps(self.best_hyperparams, indent=2)}")
        print(f"  Output directory for final model: {self.final_model_output_dir}")

    def run_final_training_and_evaluation(self):
        print(f"\n🚀 Starting Final Model Training & Test Evaluation")
        print(f"{'='*70}")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = MultiTaskBERT(dropout_rate=self.best_hyperparams['dropout']).to(device) # 'dropout' key is correct
        optimizer = AdamW(
        model.parameters(),
        lr=self.best_hyperparams['lr'], # <<<<<<<<<<<< CORRECTED KEY TO 'lr'
        weight_decay=self.best_hyperparams['weight_decay'] # 'weight_decay' key is correct
    )

        print(f"\n📊 Loading and combining train+val datasets for all tasks...")
        dataloaders_final_train = {}
        num_workers = self.best_hyperparams.get('num_workers', 0) # Get from best_hyperparams
        batch_size_final = self.best_hyperparams['batch_size']

        aux_task_batch_size_final = max(1, int(batch_size_final / 2))

        # Paths to ORIGINAL FULL CSVs
        main_original_full_path = f"{self.dataset_root_dir}/latent_hatred_3class.csv"
        stereo_original_full_path = f"{self.dataset_root_dir}/stereoset.csv"
        sarcasm_original_full_path = f"{self.dataset_root_dir}/isarcasm.csv"
        fine_original_full_path = f"{self.dataset_root_dir}/implicit_fine_labels.csv"

        # Main Task (Latent Hatred)
        # Dataset classes will handle the split. We load "train" and "val" splits separately.
        main_train_ds = LatentHatredDataset(main_original_full_path, split="train")
        main_val_ds = LatentHatredDataset(main_original_full_path, split="val")
        main_combined_ds = ConcatDataset([main_train_ds, main_val_ds])
        dataloaders_final_train["main"] = DataLoader(main_combined_ds, batch_size=batch_size_final, shuffle=True, num_workers=num_workers)

        # StereoSet Task
        stereo_train_ds = StereoSetDataset(stereo_original_full_path, split="train")
        stereo_val_ds = StereoSetDataset(stereo_original_full_path, split="val")
        stereo_combined_ds = ConcatDataset([stereo_train_ds, stereo_val_ds])
        dataloaders_final_train["stereo"] = DataLoader(stereo_combined_ds, batch_size=aux_task_batch_size_final, shuffle=True, num_workers=num_workers)
        
        # ISarcasm Task
        sarcasm_train_ds = ISarcasmDataset(sarcasm_original_full_path, split="train")
        sarcasm_val_ds = ISarcasmDataset(sarcasm_original_full_path, split="val")
        sarcasm_combined_ds = ConcatDataset([sarcasm_train_ds, sarcasm_val_ds])
        dataloaders_final_train["sarcasm"] = DataLoader(sarcasm_combined_ds, batch_size=aux_task_batch_size_final, shuffle=True, num_workers=num_workers)

        # ImplicitFineHate Task
        fine_train_ds = ImplicitFineHateDataset(fine_original_full_path, split="train")
        fine_val_ds = ImplicitFineHateDataset(fine_original_full_path, split="val")
        fine_combined_ds = ConcatDataset([fine_train_ds, fine_val_ds])
        dataloaders_final_train["implicit_fine"] = DataLoader(fine_combined_ds, batch_size=aux_task_batch_size_final, shuffle=True, num_workers=num_workers)
        print("Train+Val datasets loaded and combined.")

        task_weights = {
            "main": self.best_hyperparams['main_weight'],
            "stereo": self.best_hyperparams['stereo_weight'],
            "sarcasm": self.best_hyperparams['sarcasm_weight'],
            "implicit_fine": self.best_hyperparams['implicit_fine_weight']
        }

        # Use 'epochs_per_trial' from best_hyperparams as the number of epochs for final training
        # Or, if HPO returned a specific 'best_overall_trial_best_epoch', that could be used,
        # but often final training is done for the full duration defined by the best config.
        num_final_epochs = self.best_hyperparams['epochs'] 
        
        print(f"\n🎯 Training final model for {num_final_epochs} epochs using combined train+val data...")
        final_model_training_history = []
        for epoch in range(num_final_epochs):
            epoch_start_time = time.time()
            current_epoch_num = epoch + 1
            
            avg_train_loss = train_final_epoch_multitask(
                model, dataloaders_final_train, optimizer, task_weights, device,
                current_epoch_num, num_final_epochs, verbose_batch_logging=True
            )
            epoch_duration = time.time() - epoch_start_time
            print(f"  Final Training Epoch {current_epoch_num} Avg Weighted Loss: {avg_train_loss:.4f} (took {epoch_duration:.1f}s)")
            final_model_training_history.append({
                'epoch': current_epoch_num, 
                'avg_train_loss_on_train_val_combined': avg_train_loss,
                'epoch_duration_seconds': epoch_duration
            })
        
        final_model_save_path = self.final_model_output_dir / "final_trained_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'epochs_trained_final': num_final_epochs,
            'final_training_hyperparameters': self.best_hyperparams # Save the config it was trained with
        }, final_model_save_path)
        print(f"\n💾 Final trained model (on train+val) saved to: {final_model_save_path}")
        save_json_to_file(final_model_training_history, self.final_model_output_dir / "final_model_training_on_train_val_history.json")

        print(f"\n📊 Evaluating final model on TEST sets (used only ONCE)...")
        dataloaders_test = {}
        # Main Task (Latent Hatred) - Test
        main_test_ds = LatentHatredDataset(main_original_full_path, split="test")
        dataloaders_test["main"] = DataLoader(main_test_ds, batch_size=batch_size_final, num_workers=num_workers)
        # StereoSet Task - Test
        stereo_test_ds = StereoSetDataset(stereo_original_full_path, split="test")
        dataloaders_test["stereo"] = DataLoader(stereo_test_ds, batch_size=aux_task_batch_size_final, num_workers=num_workers)
        # ISarcasm Task - Test
        sarcasm_test_ds = ISarcasmDataset(sarcasm_original_full_path, split="test")
        dataloaders_test["sarcasm"] = DataLoader(sarcasm_test_ds, batch_size=aux_task_batch_size_final, num_workers=num_workers)
        # ImplicitFineHate Task - Test
        fine_test_ds = ImplicitFineHateDataset(fine_original_full_path, split="test")
        dataloaders_test["implicit_fine"] = DataLoader(fine_test_ds, batch_size=aux_task_batch_size_final, num_workers=num_workers)
        print("Test datasets loaded (with in-memory splitting).")
        
        final_test_metrics_all_tasks = evaluate_final_multitask_on_test(model, dataloaders_test, device)

        print(f"\n📈 Final Model Test Set Performance:")
        for task_name, metrics_dict in final_test_metrics_all_tasks.items():
            print_metrics_summary(metrics_dict, title=f"  Task: {task_name.upper()} - FINAL TEST Performance")
        
        save_json_to_file(final_test_metrics_all_tasks, self.final_model_output_dir / "final_model_test_metrics_all_tasks.json")
        
        print(f"\n🎉 Final Model Training and Test Evaluation Completed!")
        return final_test_metrics_all_tasks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Final Multi-Task Model Trainer - Standalone Runner")
    parser.add_argument("--dataset_root_dir", type=str, default="/home/avsar/codes-v2/datasets", help="Root directory of datasets")
    parser.add_argument("--final_model_output_dir", type=str, default="./final_multitask_model_run_output", help="Directory to save final model and test results")
    parser.add_argument("--best_hyperparams_json_path", type=str, required=True, help="Path to JSON file containing the best hyperparameters found by HPO.")

    args_cli = parser.parse_args()

    if not Path(args_cli.best_hyperparams_json_path).exists():
        print(f"ERROR: Best hyperparameters JSON file not found at {args_cli.best_hyperparams_json_path}")
        sys.exit(1)
    
    with open(args_cli.best_hyperparams_json_path, 'r') as f:
        best_hpo_params = json.load(f)
        # The HPO script saves the full config used for the trial, which is what we need.
        # If it saved only tunable, we'd need to merge with fixed here.
        # Assuming best_hpo_params is the 'config_used_for_trial' or 'best_overall_hpo_config'
        # which should include keys like 'dropout', 'learning_rate', 'batch_size', 'epochs_per_trial',
        # 'main_weight', 'aux_weight_stereo', etc. AND 'weight_decay', 'num_workers'.
        # Ensure all necessary keys are present.
        required_keys = ['dropout', 'lr', 'weight_decay', 'batch_size', 'num_workers', # Changed 'learning_rate' to 'lr'
                         'epochs', 
                         'main_weight', 'stereo_weight', # These are the arg names for weights
                         'sarcasm_weight', 'implicit_fine_weight'] # These are the arg names for weights
        missing_keys = [k for k in required_keys if k not in best_hpo_params]
        if missing_keys:
            print(f"ERROR: The best_hyperparams_json is missing required keys (should match HPO trial args): {missing_keys}")
            print(f"It should contain the full configuration used for the best HPO trial's args_for_training_script.")
            sys.exit(1)


    trainer_instance = FinalMultiTaskModelTrainer(
        dataset_root_dir=args_cli.dataset_root_dir,
        best_hyperparams_dict=best_hpo_params, # Pass the loaded dictionary
        final_model_output_dir=args_cli.final_model_output_dir
    )
    final_test_results = trainer_instance.run_final_training_and_evaluation()
    
    print("\n--- Standalone FinalMultiTaskModelTrainer Test Finished ---")
    if final_test_results.get("main"):
        print(f"Example final metric (Main Task F1 on Test): {final_test_results['main'].get('f1', 'N/A'):.4f}")
    print(f"All final model artifacts saved in: {args_cli.final_model_output_dir}")