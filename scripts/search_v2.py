# scripts/search_v2.py

import argparse
import itertools
import time
import pandas as pd
import torch # Keep for device, if any direct model interaction remains
import json
from pathlib import Path

import sys
import os
# Ensure project root is discoverable if running standalone or called by orchestrator
# This assumes search_v2.py is in 'scripts/' which is in the project root.
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# We will import the main training function for multi-task trials
# from scripts.train import main as train_trial_main_func # Done inside run_experiments

# For this HPO script's own evaluation (if any) or result interpretation
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score # Not strictly needed if only relying on train.py's output

# --- Constants for HPO script ---
PRIMARY_TASK_FOR_RANKING = "main"
PRIMARY_METRIC_FOR_RANKING = "f1"

def run_experiments_with_single_task_hpo_features(
    hpo_output_base_dir="hpo_multitask_runs_adapted", # Changed default from previous version
    fixed_params_for_hpo_run=None
):
    # Import here to ensure sys.path modification (if any) has taken effect
    # and to avoid circular dependencies if train.py also imported search_v2.py (it doesn't)
    from scripts.train import main as train_trial_main_func

    # --- Define Search Space ---
    """
    search_space_config = {
        "learning_rate": [3e-5],       # Example: [1e-5, 3e-5]
        "dropout": [0.2],              # Example: [0.1, 0.2, 0.3]
        "batch_size": [32],            # Example: [16, 32]
        "epochs_per_trial": [1],       # Example: [3, 5] # This is 'max_epochs'
        "main_weight": [1.0],          # Often fixed
        "aux_task_weight": [1.0]       # Example: [0.3, 0.5, 1.0] # A single weight for all aux tasks
    }
    """
    search_space_config = {
    "learning_rate": [1e-5, 2e-5, 3e-5, 5e-5],
    "dropout": [0.1],
    "batch_size": [16, 32],
    "epochs_per_trial": [1,2,3,4,5], # More epochs for real HPO
    "main_weight": [1.0],
    "aux_task_weight": [0.5, 1.0, 2.0]
    }

    # This mapping helps bridge keys from search_space_config to keys in args_for_training_script
    # which are then stored in 'config_used_for_trial'.
    search_to_arg_key_map = {
        "learning_rate": "lr",
        "dropout": "dropout", # Name matches
        "batch_size": "batch_size", # Name matches
        "epochs_per_trial": "epochs",
        "main_weight": "main_weight", # Name matches
        # If 'aux_task_weight' from search_space_config maps to multiple args (stereo_weight, sarcasm_weight etc.)
        # for summary purposes, we might pick one or list them.
        # Here, we assume it was intended that aux_task_weight applies to all, and stereo_weight can represent it.
        "aux_task_weight": "stereo_weight"
    }
    # Get the original tunable parameter keys from the search space definition
    tunable_param_keys_from_search_space = list(search_space_config.keys())


    # --- Fixed Parameters for all HPO trials for THIS HPO RUN ---
    default_fixed_params = {
        "dataset_root_dir": "/path/to/default/datasets_on_linux_placeholder", # Will be overridden by orchestrator
        "num_workers": 4,  # SET TO A POSITIVE VALUE FOR PERFORMANCE
        "weight_decay": 0.01,
        "resume_trial": False
    }
    if fixed_params_for_hpo_run: # Orchestrator can pass fixed params like the correct dataset_root_dir
        default_fixed_params.update(fixed_params_for_hpo_run)

    # --- HPO Run Setup ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    current_hpo_run_dir = Path(hpo_output_base_dir) / f"hpo_run_{timestamp}"
    current_hpo_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"💾 Hyperparameter search results will be saved in: {current_hpo_run_dir}")
    print(f"🔧 Fixed HPO Run Parameters (after potential override): {default_fixed_params}")
    print(f"📖 Tunable Search Space: {search_space_config}")

    # --- Prepare for storing HPO results ---
    all_hpo_trial_results_log = []
    best_overall_score_hpo = -float('inf')
    best_overall_config_hpo = None
    best_overall_trial_best_epoch = None
    best_overall_trial_artifact_dir = None

    # --- Generate all hyperparameter combinations for Grid Search ---
    param_names_for_product = list(search_space_config.keys())
    param_values_for_product = list(search_space_config.values())
    all_hyperparam_combinations = list(itertools.product(*param_values_for_product))
    total_experiments_to_run = len(all_hyperparam_combinations)

    print(f"\n🔬 Starting Hyperparameter Search ({total_experiments_to_run} experiments)")
    print(f"{'='*70}")

    for exp_id, current_hyperparam_combination_values in enumerate(all_hyperparam_combinations, 1):
        # Config for this specific trial (using keys from search_space_config)
        current_trial_tunable_config = dict(zip(param_names_for_product, current_hyperparam_combination_values))

        # Full config for this trial (tunable + HPO-run fixed params)
        full_config_for_this_trial = {
            **default_fixed_params,
            **current_trial_tunable_config
        }

        # Map to arg names expected by scripts.train.main
        args_for_training_script = argparse.Namespace(
            dataset_dir=full_config_for_this_trial["dataset_root_dir"],
            resume=full_config_for_this_trial["resume_trial"],
            batch_size=full_config_for_this_trial["batch_size"],
            epochs=full_config_for_this_trial["epochs_per_trial"],
            lr=full_config_for_this_trial["learning_rate"],
            dropout=full_config_for_this_trial["dropout"],
            weight_decay=full_config_for_this_trial["weight_decay"],
            num_workers=full_config_for_this_trial["num_workers"],
            main_weight=full_config_for_this_trial["main_weight"],
            stereo_weight=full_config_for_this_trial["aux_task_weight"],  # "aux_task_weight" -> "stereo_weight"
            sarcasm_weight=full_config_for_this_trial["aux_task_weight"], # "aux_task_weight" -> "sarcasm_weight"
            implicit_fine_weight=full_config_for_this_trial["aux_task_weight"] # "aux_task_weight" -> "implicit_fine_weight"
            # ... and also fixed params like "num_workers", "weight_decay", "dataset_dir", "checkpoint_dir", "resume"
        )

        trial_log_name_suffix = '_'.join([f'{k[:2]}{v}' for k,v in current_trial_tunable_config.items()])
        trial_log_name = f"Exp{exp_id:03d}_{trial_log_name_suffix}"
        trial_artifact_output_dir = current_hpo_run_dir / trial_log_name
        trial_artifact_output_dir.mkdir(parents=True, exist_ok=True)
        args_for_training_script.checkpoint_dir = str(trial_artifact_output_dir)

        print(f"\n{'='*50}")
        print(f"🚀 Experiment {exp_id}/{total_experiments_to_run}: {trial_log_name}")
        print(f"   Full Configuration for Trial (passed to train.py): {vars(args_for_training_script)}")
        print(f"   Training for {args_for_training_script.epochs} epochs. Artifacts in: {trial_artifact_output_dir}")
        print(f"{'='*50}")

        start_time_hpo_trial_processing = time.time()
        
        trial_outcome_for_hpo_log = {
            'experiment_id': exp_id,
            'trial_log_name': trial_log_name,
            # Store the args actually passed to train.py, as they include mappings and fixed params
            'config_used_for_trial': vars(args_for_training_script).copy(),
            # Also store the original tunable config for easier reference to search space
            'original_tunable_config_from_search_space': current_trial_tunable_config.copy(),
            'trial_artifact_dir': str(trial_artifact_output_dir),
            'success': False,
        }

        try:
            train_trial_main_func(args_for_training_script, trial_log_name_from_hpo=trial_log_name)
            trial_outcome_for_hpo_log['training_execution_time_seconds'] = time.time() - start_time_hpo_trial_processing
            print(f"   ✅ Training script execution completed for {trial_log_name}.")

            best_metrics_summary_file_from_trial = trial_artifact_output_dir / "best_metrics_summary_for_trial.json"
            if not best_metrics_summary_file_from_trial.exists():
                raise FileNotFoundError(f"'best_metrics_summary_for_trial.json' not found in {trial_artifact_output_dir}.")

            with open(best_metrics_summary_file_from_trial, 'r') as f_bm:
                trial_best_summary_data = json.load(f_bm)
            
            trial_outcome_for_hpo_log['best_epoch_reported_by_trial'] = trial_best_summary_data.get("best_epoch_for_this_trial")
            metrics_from_trial_best_epoch = trial_best_summary_data.get("metrics_all_tasks_at_best_epoch", {})
            trial_outcome_for_hpo_log['metrics_from_trial_internal_best_epoch'] = metrics_from_trial_best_epoch
            trial_outcome_for_hpo_log['success'] = True

            score_reported_by_trial_for_ranking = metrics_from_trial_best_epoch.get(PRIMARY_TASK_FOR_RANKING, {}).get(PRIMARY_METRIC_FOR_RANKING, -float('inf'))
            trial_outcome_for_hpo_log['primary_score_for_hpo_ranking'] = score_reported_by_trial_for_ranking
            
            print(f"   📊 Score reported by trial for HPO ranking ({PRIMARY_TASK_FOR_RANKING} {PRIMARY_METRIC_FOR_RANKING}): {score_reported_by_trial_for_ranking:.4f}")
            if trial_outcome_for_hpo_log['best_epoch_reported_by_trial'] is not None:
                 print(f"   (This score is from {PRIMARY_TASK_FOR_RANKING}'s validation during the trial at its epoch {trial_outcome_for_hpo_log['best_epoch_reported_by_trial']})")

        except Exception as e:
            print(f"   ❌ Trial {exp_id} ({trial_log_name}) failed: {e}")
            import traceback
            traceback.print_exc()
            trial_outcome_for_hpo_log['success'] = False
            trial_outcome_for_hpo_log['error_message'] = str(e)
            trial_outcome_for_hpo_log['primary_score_for_hpo_ranking'] = -float('inf')

        trial_outcome_for_hpo_log['hpo_total_trial_processing_time_seconds'] = time.time() - start_time_hpo_trial_processing
        all_hpo_trial_results_log.append(trial_outcome_for_hpo_log)

        if trial_outcome_for_hpo_log['success'] and \
           trial_outcome_for_hpo_log['primary_score_for_hpo_ranking'] > best_overall_score_hpo:
            best_overall_score_hpo = trial_outcome_for_hpo_log['primary_score_for_hpo_ranking']
            best_overall_config_hpo = trial_outcome_for_hpo_log['config_used_for_trial'].copy() # Store the args version
            best_overall_trial_best_epoch = trial_outcome_for_hpo_log['best_epoch_reported_by_trial']
            best_overall_trial_artifact_dir = trial_outcome_for_hpo_log['trial_artifact_dir']
            
            print(f"   🎯 NEW OVERALL BEST HPO CONFIGURATION FOUND!")
            print(f"      Score ({PRIMARY_TASK_FOR_RANKING} {PRIMARY_METRIC_FOR_RANKING}): {best_overall_score_hpo:.4f}")
            if best_overall_trial_best_epoch is not None:
                print(f"      Achieved at Trial's Best Epoch: {best_overall_trial_best_epoch}")
            print(f"      Full Config (as passed to train.py): {best_overall_config_hpo}")
            print(f"      Trial Artifacts: {best_overall_trial_artifact_dir}")

        with open(current_hpo_run_dir / "hpo_all_trials_log.json", 'w') as f_json:
            json.dump(all_hpo_trial_results_log, f_json, indent=2)
        
        if all_hpo_trial_results_log:
            df_summary_data = []
            for r_log in all_hpo_trial_results_log:
                row = r_log['original_tunable_config_from_search_space'].copy() # Start with original tunable for CSV
                row.update(default_fixed_params) # Add fixed HPO params for full context, ensuring no overwrite of tunables
                row['experiment_id'] = r_log['experiment_id']
                row['trial_log_name'] = r_log['trial_log_name']
                row['success'] = r_log['success']
                row['primary_score_for_hpo_ranking'] = r_log.get('primary_score_for_hpo_ranking')
                row['best_epoch_reported_by_trial'] = r_log.get('best_epoch_reported_by_trial')
                row['training_execution_time_seconds'] = r_log.get('training_execution_time_seconds')
                trial_best_metrics = r_log.get('metrics_from_trial_internal_best_epoch', {})
                primary_task_metrics = trial_best_metrics.get(PRIMARY_TASK_FOR_RANKING, {})
                for metric_key, metric_val in primary_task_metrics.items():
                    if isinstance(metric_val, (int, float)):
                         row[f'trial_best_{PRIMARY_TASK_FOR_RANKING}_{metric_key}'] = metric_val
                row['error_message'] = r_log.get('error_message', '')
                df_summary_data.append(row)
            df_hpo = pd.DataFrame(df_summary_data)
            df_hpo.to_csv(current_hpo_run_dir / "hpo_summary_log.csv", index=False)
        print(f"   Intermediate HPO log files saved to {current_hpo_run_dir}")

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
            print(f"\n🏆 OVERALL BEST HPO CONFIGURATION (as passed to train.py):")
            print(f"   Score ({PRIMARY_TASK_FOR_RANKING} {PRIMARY_METRIC_FOR_RANKING}): {best_overall_score_hpo:.4f}")
            if best_overall_trial_best_epoch is not None:
                print(f"   Achieved at Trial's Best Epoch: {best_overall_trial_best_epoch}")
            print(f"   Full Configuration:")
            for param_name, param_val in best_overall_config_hpo.items(): # best_overall_config_hpo has arg names
                print(f"     {param_name}: {param_val}")
            print(f"   Artifacts from best trial are in: {best_overall_trial_artifact_dir}")
            print(f"   To use this best model, load 'checkpoint_best_model.pt' from that directory.")
        else:
            print("\n   No best HPO configuration found among successful trials.")

        sorted_successful_trials = sorted(
            successful_hpo_trials,
            key=lambda x: x.get('primary_score_for_hpo_ranking', -float('inf')),
            reverse=True
        )
        print(f"\n📈 TOP 5 HPO CONFIGURATIONS (based on {PRIMARY_TASK_FOR_RANKING} {PRIMARY_METRIC_FOR_RANKING} reported by trial):")
        for i, top_trial_log in enumerate(sorted_successful_trials[:5], 1):
            score = top_trial_log['primary_score_for_hpo_ranking']
            # Use original_tunable_config_from_search_space for displaying what was "tuned"
            config_tuned_display = top_trial_log['original_tunable_config_from_search_space']
            epoch_val = top_trial_log['best_epoch_reported_by_trial']
            primary_task_mets = top_trial_log.get('metrics_from_trial_internal_best_epoch', {}).get(PRIMARY_TASK_FOR_RANKING, {})
            acc = primary_task_mets.get('accuracy', 'N/A')
            prec = primary_task_mets.get('precision', 'N/A')
            rec = primary_task_mets.get('recall', 'N/A')

            print(f"   {i}. Score={score:.4f} (Trial's Best Epoch: {epoch_val})")
            print(f"      Acc={acc if isinstance(acc,str) else f'{acc:.4f}'}, P={prec if isinstance(prec,str) else f'{prec:.4f}'}, R={rec if isinstance(rec,str) else f'{rec:.4f}'} (for {PRIMARY_TASK_FOR_RANKING})")
            print(f"      Config (Tunable from Search Space):")
            for k_tuned, v_tuned in config_tuned_display.items():
                 print(f"         {k_tuned}: {v_tuned}")

        print(f"\n🔍 PARAMETER IMPACT ANALYSIS (on {PRIMARY_TASK_FOR_RANKING} {PRIMARY_METRIC_FOR_RANKING} reported by trial):")
        df_for_analysis_data = []
        for r in successful_hpo_trials:
            row_data = r['original_tunable_config_from_search_space'].copy()
            row_data['score_to_analyze'] = r['primary_score_for_hpo_ranking']
            df_for_analysis_data.append(row_data)
        df_for_analysis = pd.DataFrame(df_for_analysis_data)
        
        # tunable_param_keys_from_search_space are the keys from search_space_config
        for param_key in tunable_param_keys_from_search_space:
            if param_key in df_for_analysis.columns:
                try:
                    param_analysis_stats = df_for_analysis.groupby(param_key)['score_to_analyze'].agg(['mean', 'std', 'count']).sort_values(by='mean', ascending=False)
                    print(f"\n   --- {param_key.upper()} ---")
                    for val_level, stats_row in param_analysis_stats.iterrows(): # Use stats_row to avoid conflict
                        print(f"     {val_level}: Mean_Score={stats_row['mean']:.4f} ± {stats_row['std']:.4f} (n={int(stats_row['count'])})")
                except Exception as e_param:
                    print(f"     Could not analyze parameter {param_key}: {e_param}")

    final_hpo_analysis_data = {
        'best_overall_hpo_config_as_args': best_overall_config_hpo, # This is vars(args_for_training_script)
        'best_overall_score_hpo': best_overall_score_hpo,
        'best_overall_trial_best_epoch': best_overall_trial_best_epoch,
        'best_overall_trial_artifact_dir': best_overall_trial_artifact_dir,
        'hpo_fixed_parameters': default_fixed_params, # These are the ones active for the HPO run
        'hpo_search_space_definition': search_space_config, # The original search space tried
        'primary_ranking_info': {'task': PRIMARY_TASK_FOR_RANKING, 'metric': PRIMARY_METRIC_FOR_RANKING},
        'top_5_hpo_configurations_summary': [
            {
                'rank': i+1,
                'score': t['primary_score_for_hpo_ranking'],
                'best_epoch_in_trial': t['best_epoch_reported_by_trial'],
                # Store the original search space config for this trial for clarity
                'config_tunable_from_search_space': t['original_tunable_config_from_search_space'],
                'full_config_as_args_for_trial': t['config_used_for_trial'], # For full reproducibility
                'trial_artifact_dir': t['trial_artifact_dir']
            } for i, t in enumerate(sorted_successful_trials[:5])
        ] if successful_hpo_trials else [],
        'total_hpo_experiments_run': total_experiments_to_run,
        'num_successful_hpo_trials': len(successful_hpo_trials)
    }
    with open(current_hpo_run_dir / "hpo_final_analysis_and_summary.json", 'w') as f_analysis:
        json.dump(final_hpo_analysis_data, f_analysis, indent=2, default=str) # default=str for Path objects etc.

    print(f"\n💾 Final HPO analysis and summary saved to: {current_hpo_run_dir}")
    print(f"{'='*70}")

    return best_overall_config_hpo, best_overall_score_hpo, best_overall_trial_best_epoch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Multi-Task Hyperparameter Experiments (Functional, Single-Task HPO Style)")
    parser.add_argument(
        "--hpo_output_base_dir", type=str, default="hpo_multitask_runs_cli_test",
        help="Base directory to save all HPO run results."
    )
    parser.add_argument(
        "--dataset_root_dir_override", type=str, default=None,
        help="Override the dataset_root_dir for this HPO run (e.g., './datasets')."
    )
    parser.add_argument(
        "--num_workers_override", type=int, default=None,
        help="Override the num_workers for this HPO run."
    )
    
    cli_hpo_args = parser.parse_args()

    try:
        from scripts.train import main as train_main_check # Check import again here for safety
        print("Successfully imported 'main' from 'scripts.train'. This will be called for each trial.")
    except ImportError as e:
        print(f"ERROR: Could not import 'main' from 'scripts.train'. {e}")
        print("Ensure project root (e.g., '~/hate-speech-detection') is in PYTHONPATH, and 'scripts/train.py' with a 'main(args, trial_log_name_from_hpo)' function exists.")
        sys.exit(1)
    
    # Prepare fixed params that might be overridden by CLI
    hpo_run_fixed_params = {}
    if cli_hpo_args.dataset_root_dir_override:
        hpo_run_fixed_params["dataset_root_dir"] = cli_hpo_args.dataset_root_dir_override
    if cli_hpo_args.num_workers_override is not None: # Check for None as 0 is a valid value
        hpo_run_fixed_params["num_workers"] = cli_hpo_args.num_workers_override

    run_experiments_with_single_task_hpo_features(
        hpo_output_base_dir=cli_hpo_args.hpo_output_base_dir,
        fixed_params_for_hpo_run=hpo_run_fixed_params if hpo_run_fixed_params else None
    )