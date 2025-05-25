# search_v2.py (Function-based HPO, adapted with Single-Task HyperparameterSearchManager features)

import argparse
import itertools
import time
import pandas as pd
import torch # Keep for device, if any direct model interaction remains (though less now)
import json
from pathlib import Path

import sys
import os
sys.path.append(os.path.expanduser("~/codes-v2")) # Ensure your project root is in path

# We will import the main training function for multi-task trials
# from scripts.train import main as train_trial_main_func # Done inside run_experiments

# For this HPO script's own evaluation (if any) or result interpretation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# from models.multitask_bert import MultiTaskBERT # Not needed if HPO only reads JSON results
# from scripts.dataset_loaders import LatentHatredDataset, StereoSetDataset, ISarcasmDataset, ImplicitFineHateDataset # Not needed if HPO only reads JSON results
# from torch.utils.data import DataLoader # Not needed if HPO only reads JSON results

# --- Constants for HPO script ---
# These should align with what scripts.train.py uses as its primary target for "best"
PRIMARY_TASK_FOR_RANKING = "main"
PRIMARY_METRIC_FOR_RANKING = "f1"

def run_experiments_with_single_task_hpo_features(
    hpo_output_base_dir="hpo_multitask_runs_final_adapt",
    # Parameters that might be fixed for this HPO run
    fixed_params_for_hpo_run=None
):
    from scripts.train import main as train_trial_main_func # Import here

    # --- Define Search Space (similar to HyperparameterSearchManager.define_search_space) ---
    search_space_config = {
        "learning_rate": [3e-5], #[1e-5, 2e-5, 3e-5, 5e-5],
        "dropout": [0.2, 0.3],
        "batch_size": [8, 16],
        "epochs_per_trial": [3, 4, 5], # This is 'max_epochs' from single-task HPO
        "main_weight": [1.0], # Often fixed
        "aux_task_weight": [0.3, 0.5, 1.0] # A single weight for all aux tasks
    }
    # Combine with any fixed parameters for the HPO run
    # These fixed params are analogous to self.fixed_dropout etc. in single-task HPO
    default_fixed_params = {
        "dataset_root_dir": "/home/avsar/codes-v2/datasets", # Or get from args
        "num_workers": 0,
        "weight_decay": 0.01,
        "resume_trial": False # Usually False for HPO trials
    }
    if fixed_params_for_hpo_run:
        default_fixed_params.update(fixed_params_for_hpo_run)


    # --- HPO Run Setup ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_hpo_run_dir = Path(hpo_output_base_dir) / f"hpo_run_{timestamp}"
    current_hpo_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Hyperparameter search results will be saved in: {current_hpo_run_dir}")
    print(f"🔧 Fixed HPO Run Parameters: {default_fixed_params}")
    print(f"📖 Tunable Search Space: {search_space_config}")

    # --- Prepare for storing results (mimicking HyperparameterSearchManager) ---
    all_hpo_trial_results_log = [] # Like self.all_results
    best_overall_score_hpo = -float('inf') # Like self.best_score
    best_overall_config_hpo = None   # Like self.best_config
    best_overall_trial_best_epoch = None # Like self.best_epoch (from the best trial)
    best_overall_trial_artifact_dir = None


    # --- Generate all combinations for HPO (Grid Search) ---
    param_names = list(search_space_config.keys())
    param_values = list(search_space_config.values())
    all_hyperparam_combinations = list(itertools.product(*param_values))
    total_experiments_to_run = len(all_hyperparam_combinations)

    print(f"\n🔬 Starting Hyperparameter Search ({total_experiments_to_run} experiments)")
    print(f"{'='*70}")

    for exp_id, current_hyperparam_combination in enumerate(all_hyperparam_combinations, 1):
        # Create config for this specific trial (tunable params)
        current_trial_tunable_config = dict(zip(param_names, current_hyperparam_combination))

        # Full config for this trial (tunable + HPO-run fixed params)
        full_config_for_this_trial = {
            **default_fixed_params, # Start with HPO-run fixed params
            **current_trial_tunable_config # Add/override with current trial's tunable params
        }
        # Specifically map to arg names expected by scripts.train.main
        # This is like HyperparameterSearchManager.train_single_config creating 'args'
        args_for_training_script = argparse.Namespace(
            dataset_dir=full_config_for_this_trial["dataset_root_dir"],
            # checkpoint_dir will be set below, unique for the trial
            resume=full_config_for_this_trial["resume_trial"],
            batch_size=full_config_for_this_trial["batch_size"],
            epochs=full_config_for_this_trial["epochs_per_trial"], # This is key
            lr=full_config_for_this_trial["learning_rate"],
            dropout=full_config_for_this_trial["dropout"],
            weight_decay=full_config_for_this_trial["weight_decay"],
            num_workers=full_config_for_this_trial["num_workers"],
            main_weight=full_config_for_this_trial["main_weight"],
            stereo_weight=full_config_for_this_trial["aux_task_weight"], # Using one aux weight for all
            sarcasm_weight=full_config_for_this_trial["aux_task_weight"],
            implicit_fine_weight=full_config_for_this_trial["aux_task_weight"]
        )

        trial_log_name = f"Exp{exp_id:03d}_{'_'.join([f'{k[:2]}{v}' for k,v in current_trial_tunable_config.items()])}"
        trial_artifact_output_dir = current_hpo_run_dir / trial_log_name
        trial_artifact_output_dir.mkdir(parents=True, exist_ok=True)
        args_for_training_script.checkpoint_dir = str(trial_artifact_output_dir) # Set unique dir

        print(f"\n{'='*50}")
        print(f"🚀 Experiment {exp_id}/{total_experiments_to_run}: {trial_log_name}")
        print(f"   Full Configuration for Trial: {vars(args_for_training_script)}")
        print(f"   Training for {args_for_training_script.epochs} epochs. Artifacts in: {trial_artifact_output_dir}")
        print(f"{'='*50}")

        start_time_hpo_trial_processing = time.time()
        
        # This structure mimics HyperparameterSearchManager.train_single_config
        # It calls the training script and then processes its results.
        trial_outcome_for_hpo_log = {
            'experiment_id': exp_id,
            'trial_log_name': trial_log_name,
            'config_used_for_trial': vars(args_for_training_script).copy(), # Log the exact args passed
            'trial_artifact_dir': str(trial_artifact_output_dir),
            'success': False, # Default to False
        }

        try:
            # --- Call the enhanced scripts.train.main ---
            # This function will run the trial, do internal validation, and save its
            # own 'best_model_checkpoint.pt' and 'best_metrics_summary_for_trial.json'.
            train_trial_main_func(args_for_training_script, trial_log_name_from_hpo=trial_log_name)
            trial_outcome_for_hpo_log['training_execution_time_seconds'] = time.time() - start_time_hpo_trial_processing
            print(f"   ✅ Training script execution completed for {trial_log_name}.")

            # --- HPO: Read the best metrics reported BY THE TRIAL ITSELF ---
            # This is like HyperparameterSearchManager reading results['checkpoint_manager'] / "best_metrics.json"
            best_metrics_summary_file_from_trial = trial_artifact_output_dir / "best_metrics_summary_for_trial.json"
            if not best_metrics_summary_file_from_trial.exists():
                raise FileNotFoundError(f"'best_metrics_summary_for_trial.json' not found in {trial_artifact_output_dir}. Training script might have failed to produce it.")

            with open(best_metrics_summary_file_from_trial, 'r') as f_bm:
                trial_best_summary_data = json.load(f_bm)
            
            trial_outcome_for_hpo_log['best_epoch_reported_by_trial'] = trial_best_summary_data.get("best_epoch_for_this_trial")
            metrics_from_trial_best_epoch = trial_best_summary_data.get("metrics_all_tasks_at_best_epoch", {})
            trial_outcome_for_hpo_log['metrics_from_trial_internal_best_epoch'] = metrics_from_trial_best_epoch # Store all task metrics
            trial_outcome_for_hpo_log['success'] = True

            # Extract the primary metric value reported by the trial for HPO ranking
            # This is like HyperparameterSearchManager getting results['best_val_f1']
            score_reported_by_trial_for_ranking = metrics_from_trial_best_epoch.get(PRIMARY_TASK_FOR_RANKING, {}).get(PRIMARY_METRIC_FOR_RANKING, -float('inf'))
            trial_outcome_for_hpo_log['primary_score_for_hpo_ranking'] = score_reported_by_trial_for_ranking
            
            print(f"   📊 Score reported by trial for HPO ranking ({PRIMARY_TASK_FOR_RANKING} {PRIMARY_METRIC_FOR_RANKING}): {score_reported_by_trial_for_ranking:.4f}")
            print(f"   (This score is from {PRIMARY_TASK_FOR_RANKING}'s validation during the trial at its epoch {trial_outcome_for_hpo_log['best_epoch_reported_by_trial']})")

        except Exception as e:
            print(f"   ❌ Trial {exp_id} ({trial_log_name}) failed: {e}")
            import traceback
            traceback.print_exc()
            trial_outcome_for_hpo_log['success'] = False
            trial_outcome_for_hpo_log['error_message'] = str(e)
            trial_outcome_for_hpo_log['primary_score_for_hpo_ranking'] = -float('inf') # Ensure it's low

        trial_outcome_for_hpo_log['hpo_total_trial_processing_time_seconds'] = time.time() - start_time_hpo_trial_processing
        all_hpo_trial_results_log.append(trial_outcome_for_hpo_log)

        # --- Check for new best HPO configuration (mimicking HyperparameterSearchManager) ---
        if trial_outcome_for_hpo_log['success'] and \
           trial_outcome_for_hpo_log['primary_score_for_hpo_ranking'] > best_overall_score_hpo:
            best_overall_score_hpo = trial_outcome_for_hpo_log['primary_score_for_hpo_ranking']
            # Store the 'config_used_for_trial' as it contains all params (fixed + tuned)
            best_overall_config_hpo = trial_outcome_for_hpo_log['config_used_for_trial'].copy()
            best_overall_trial_best_epoch = trial_outcome_for_hpo_log['best_epoch_reported_by_trial']
            best_overall_trial_artifact_dir = trial_outcome_for_hpo_log['trial_artifact_dir']
            
            print(f"   🎯 NEW OVERALL BEST HPO CONFIGURATION FOUND!")
            print(f"      Score ({PRIMARY_TASK_FOR_RANKING} {PRIMARY_METRIC_FOR_RANKING}): {best_overall_score_hpo:.4f}")
            if best_overall_trial_best_epoch is not None:
                print(f"      Achieved at Trial's Best Epoch: {best_overall_trial_best_epoch}")
            print(f"      Full Config: {best_overall_config_hpo}")
            print(f"      Trial Artifacts: {best_overall_trial_artifact_dir}")

        # --- Save Intermediate HPO Log Files (mimicking HyperparameterSearchManager.save_results) ---
        # Detailed JSON log for all trials run so far
        with open(current_hpo_run_dir / "hpo_all_trials_log.json", 'w') as f_json:
            json.dump(all_hpo_trial_results_log, f_json, indent=2)
        
        # Summary CSV log
        if all_hpo_trial_results_log:
            df_summary_data = []
            for r_log in all_hpo_trial_results_log:
                row = r_log['config_used_for_trial'].copy() # Start with full config
                row['experiment_id'] = r_log['experiment_id']
                row['trial_log_name'] = r_log['trial_log_name']
                row['success'] = r_log['success']
                row['primary_score_for_hpo_ranking'] = r_log.get('primary_score_for_hpo_ranking')
                row['best_epoch_reported_by_trial'] = r_log.get('best_epoch_reported_by_trial')
                row['training_execution_time_seconds'] = r_log.get('training_execution_time_seconds')
                # Add key metrics for primary task from trial's best epoch to CSV
                trial_best_metrics = r_log.get('metrics_from_trial_internal_best_epoch', {})
                primary_task_metrics = trial_best_metrics.get(PRIMARY_TASK_FOR_RANKING, {})
                for metric_key, metric_val in primary_task_metrics.items():
                    if isinstance(metric_val, (int, float)): # Only add simple numeric metrics to CSV
                         row[f'trial_best_{PRIMARY_TASK_FOR_RANKING}_{metric_key}'] = metric_val
                row['error_message'] = r_log.get('error_message', '')
                df_summary_data.append(row)
            df_hpo = pd.DataFrame(df_summary_data)
            df_hpo.to_csv(current_hpo_run_dir / "hpo_summary_log.csv", index=False)
        print(f"   Intermediate HPO log files saved to {current_hpo_run_dir}")

    # --- Final HPO Analysis (mimicking HyperparameterSearchManager.analyze_results) ---
    print(f"\n{'='*70}")
    print("✨ Hyperparameter Search Completed ✨")
    print(f"Total HPO experiments run: {len(all_hpo_trial_results_log)}")
    successful_hpo_trials = [r for r in all_hpo_trial_results_log if r['success']]
    print(f"Successful HPO trials: {len(successful_hpo_trials)}")
    print(f"Failed HPO trials: {len(all_hpo_trial_results_log) - len(successful_hpo_trials)}")

    if not successful_hpo_trials:
        print("\n❌ No successful HPO trials to analyze.")
    else:
        if best_overall_config_hpo:
            print(f"\n🏆 OVERALL BEST HPO CONFIGURATION:")
            print(f"   Score ({PRIMARY_TASK_FOR_RANKING} {PRIMARY_METRIC_FOR_RANKING}): {best_overall_score_hpo:.4f}")
            if best_overall_trial_best_epoch is not None:
                print(f"   Achieved at Trial's Best Epoch: {best_overall_trial_best_epoch}")
            print(f"   Full Configuration:")
            for param_name, param_val in best_overall_config_hpo.items():
                print(f"     {param_name}: {param_val}")
            print(f"   Artifacts from best trial are in: {best_overall_trial_artifact_dir}")
            print(f"   To use this best model, load 'checkpoint_best_model.pt' from that directory.")
        else:
            print("\n   No best HPO configuration found among successful trials.")

        # Top N configurations (mimicking HyperparameterSearchManager)
        # Sort by 'primary_score_for_hpo_ranking'
        sorted_successful_trials = sorted(
            successful_hpo_trials,
            key=lambda x: x.get('primary_score_for_hpo_ranking', -float('inf')),
            reverse=True
        )
        print(f"\n📈 TOP 5 HPO CONFIGURATIONS (based on {PRIMARY_TASK_FOR_RANKING} {PRIMARY_METRIC_FOR_RANKING} reported by trial):")
        for i, top_trial_log in enumerate(sorted_successful_trials[:5], 1):
            score = top_trial_log['primary_score_for_hpo_ranking']
            config = top_trial_log['config_used_for_trial']
            epoch = top_trial_log['best_epoch_reported_by_trial']
            # Display primary task's detailed metrics from its best epoch
            primary_task_mets = top_trial_log.get('metrics_from_trial_internal_best_epoch', {}).get(PRIMARY_TASK_FOR_RANKING, {})
            acc = primary_task_mets.get('accuracy', 'N/A')
            prec = primary_task_mets.get('precision', 'N/A') # Macro precision
            rec = primary_task_mets.get('recall', 'N/A')    # Macro recall

            print(f"   {i}. Score={score:.4f} (Trial's Best Epoch: {epoch})")
            print(f"      Acc={acc if isinstance(acc,str) else f'{acc:.4f}'}, P={prec if isinstance(prec,str) else f'{prec:.4f}'}, R={rec if isinstance(rec,str) else f'{rec:.4f}'} (for {PRIMARY_TASK_FOR_RANKING})")
            print(f"      Config (Selected Tuned):")
            for k_tuned in current_trial_tunable_config.keys(): # Show only tuned params for brevity
                 print(f"         {k_tuned}: {config.get(k_tuned)}")
            # print(f"      Full Config: {config}") # Or print full config

        # Parameter Importance Analysis (mimicking HyperparameterSearchManager)
        print(f"\n🔍 PARAMETER IMPACT ANALYSIS (on {PRIMARY_TASK_FOR_RANKING} {PRIMARY_METRIC_FOR_RANKING} reported by trial):")
        df_for_analysis = pd.DataFrame([
            {**r['config_used_for_trial'], 'score_to_analyze': r['primary_score_for_hpo_ranking']}
            for r in successful_hpo_trials
        ])
        
        tunable_param_keys = list(search_space_config.keys()) # Parameters that were varied
        for param_key in tunable_param_keys:
            if param_key in df_for_analysis.columns:
                try:
                    param_analysis_stats = df_for_analysis.groupby(param_key)['score_to_analyze'].agg(['mean', 'std', 'count']).sort_values(by='mean', ascending=False)
                    print(f"\n   --- {param_key.upper()} ---")
                    for val_level, stats in param_analysis_stats.iterrows():
                        print(f"     {val_level}: Mean_Score={stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['count']})")
                except Exception as e_param:
                    print(f"     Could not analyze parameter {param_key}: {e_param}")


    # Save final HPO analysis summary file (mimicking HyperparameterSearchManager)
    final_hpo_analysis_data = {
        'best_overall_hpo_config': best_overall_config_hpo,
        'best_overall_score_hpo': best_overall_score_hpo,
        'best_overall_trial_best_epoch': best_overall_trial_best_epoch,
        'best_overall_trial_artifact_dir': best_overall_trial_artifact_dir,
        'hpo_fixed_parameters': default_fixed_params,
        'hpo_search_space': search_space_config,
        'primary_ranking_info': {'task': PRIMARY_TASK_FOR_RANKING, 'metric': PRIMARY_METRIC_FOR_RANKING},
        'top_5_hpo_configurations_summary': [
            {
                'rank': i+1,
                'score': t['primary_score_for_hpo_ranking'],
                'best_epoch_in_trial': t['best_epoch_reported_by_trial'],
                'config_tunable': {k: t['config_used_for_trial'][k] for k in tunable_param_keys},
                'trial_artifact_dir': t['trial_artifact_dir']
            } for i, t in enumerate(sorted_successful_trials[:5])
        ] if successful_hpo_trials else [],
        'total_hpo_experiments_run': total_experiments_to_run,
        'num_successful_hpo_trials': len(successful_hpo_trials)
    }
    with open(current_hpo_run_dir / "hpo_final_analysis_and_summary.json", 'w') as f_analysis:
        json.dump(final_hpo_analysis_data, f_analysis, indent=2)

    print(f"\n💾 Final HPO analysis and summary saved to: {current_hpo_run_dir}")
    print(f"{'='*70}")

    # Return best config and score, like HyperparameterSearchManager
    return best_overall_config_hpo, best_overall_score_hpo, best_overall_trial_best_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multi-Task Hyperparameter Experiments (Functional, Single-Task HPO Style)")
    parser.add_argument(
        "--hpo_output_base_dir", type=str, default="hpo_multitask_runs_adapted",
        help="Base directory to save all HPO run results."
    )
    # Add CLI args for fixed_params_for_hpo_run if desired (e.g., --dataset_root_dir)
    
    cli_hpo_args = parser.parse_args()

    # Check importability of the training script's main function
    try:
        from scripts.train import main as train_main_check
        print("Successfully imported 'main' from 'scripts.train'. This will be called for each trial.")
    except ImportError as e:
        print(f"ERROR: Could not import 'main' from 'scripts.train'. {e}")
        print("Ensure '~/codes-v2' (or your project root) is in PYTHONPATH, and 'scripts/train.py' with a 'main(args, trial_log_name_from_hpo)' function exists.")
        sys.exit(1)
    
    # Run the HPO experiments
    run_experiments_with_single_task_hpo_features(
        hpo_output_base_dir=cli_hpo_args.hpo_output_base_dir
    )