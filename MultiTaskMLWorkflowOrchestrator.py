# MultiTaskMLWorkflowOrchestrator.py
import argparse
import os
import sys
import json
import time
from pathlib import Path

# --- Adjust imports based on your project structure ---
# Option 1: If search_v2.py and final_model_training_multitask.py are in PYTHONPATH
# from search_v2 import run_experiments_with_single_task_hpo_features
# from final_model_training_multitask import FinalMultiTaskModelTrainer

# Option 2: If they are in a 'scripts' subdirectory relative to this orchestrator
# Assuming orchestrator is in project root, and others are in 'scripts/'
try:
    # This assumes search_v2.py is directly importable (e.g. in project root or scripts dir that's in PYTHONPATH)
    # and it has the function run_experiments_with_single_task_hpo_features
    # You might need to rename your search_v2.py's main HPO function or the file itself
    # For this example, let's assume search_v2.py is in the same dir or PYTHONPATH
    # and its main HPO running function is called 'run_hpo_for_multitask'
    
    # Let's assume your search_v2.py (the one with HPO logic) is named search_v2_hpo_runner.py
    # and its main function is run_experiments_with_single_task_hpo_features
    # And final_model_training_multitask.py is in the same directory or accessible
    
    # For this example, I'll assume they are in a 'scripts' subfolder
    # and this orchestrator is one level above 'scripts' or 'scripts' is added to path.
    current_dir = Path(__file__).resolve().parent
    # If 'scripts' is a sibling to this file's directory, or this file is in 'scripts' parent.
    # This path adjustment is tricky and depends on where you place the orchestrator.
    # A common practice is to make your 'scripts' and 'models' installable packages.
    
    # Simple approach: assume they can be imported because project root is in PYTHONPATH
    # (e.g. if you run `python -m workflows.MultiTaskMLWorkflowOrchestrator` from project root)
    # OR ensure search_v2.py and final_model_training_multitask.py are in the same dir as this
    # OR add their location to sys.path here.

    # To make it runnable if placed in project root and others are in 'scripts':
    if str(current_dir / 'scripts') not in sys.path and (current_dir / 'scripts').exists():
         sys.path.insert(0, str(current_dir / 'scripts')) # Add scripts to path temporarily

    # The search_v2.py script you want to adapt (the one containing run_experiments_with_single_task_hpo_features)
    # Let's assume it's directly importable as 'search_v2' for clarity
    import scripts.search_v2 as search_v2  # Rename your search_v2.py to search_v2.py or adjust
    from scripts.final_model_training_multitask import FinalMultiTaskModelTrainer

except ImportError as e:
    print(f"ImportError: {e}. Ensure search_v2.py (with HPO logic) and final_model_training_multitask.py are accessible.")
    print("You may need to adjust sys.path or run this script from your project's root directory.")
    sys.exit(1)


# (Optional) Data Verification step - Placeholder
def verify_all_task_datasets_placeholder(dataset_root_dir_str):
    print(f"  (Data Verification Placeholder) Verifying datasets in {dataset_root_dir_str}...")
    # TODO: Implement actual checks for each task's train, val, test files
    # e.g., existence, non-empty, basic row counts, stratification per task if needed.
    # For now, return mock statistics.
    mock_stats = {
        "main_task_samples": {"train": 1000, "val": 200, "test": 200},
        "stereo_task_samples": {"train": 500, "val": 100, "test": 100},
        "sarcasm_task_samples": {"train": 800, "val": 150, "test": 150},
        "implicit_fine_task_samples": {"train": 600, "val": 120, "test": 120},
        "verification_status": "Mock verification passed (implement actual checks)."
    }
    time.sleep(0.5) # Simulate work
    return mock_stats


class MultiTaskMLWorkflowOrchestrator:
    def __init__(self, dataset_root_dir_str, base_results_dir_str="ml_workflow_multitask_runs_orchestrated"):
        self.dataset_root_dir = Path(dataset_root_dir_str)
        self.base_results_dir = Path(base_results_dir_str)
        self.base_results_dir.mkdir(parents=True, exist_ok=True)

        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.workflow_run_dir = self.base_results_dir / f"workflow_multitask_{self.timestamp}"
        self.workflow_run_dir.mkdir(exist_ok=True)

        print(f"🚀 Multi-Task ML Workflow Orchestrator Initialized")
        print(f"{'='*70}")
        print(f"Dataset Root: {self.dataset_root_dir}")
        print(f"Workflow Results Directory: {self.workflow_run_dir}")
        print(f"{'='*70}")

        self.workflow_state = {
            'start_time_unix': time.time(),
            'start_time_str': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())),
            'dataset_root_dir': str(self.dataset_root_dir),
            'workflow_run_dir': str(self.workflow_run_dir),
            'timestamp': self.timestamp,
            'steps_completed_successfully': [],
            'current_step_being_executed': None,
            'hpo_summary_results': None, # Store summary from HPO
            'final_model_test_metrics_all_tasks': None
        }
        self._save_workflow_state() # Initial state

    def _log_step_status(self, step_name, status_message):
        print(f"  [{time.strftime('%H:%M:%S')}] {step_name}: {status_message}")

    def _start_step(self, step_name):
        self._log_step_status(step_name, "Started...")
        self.workflow_state['current_step_being_executed'] = step_name
        self._save_workflow_state()

    def _complete_step(self, step_name):
        self.workflow_state['steps_completed_successfully'].append(step_name)
        self.workflow_state['current_step_being_executed'] = None # Clear current step
        self._log_step_status(step_name, "Completed Successfully.")
        self._save_workflow_state()

    def _save_workflow_state(self):
        state_file = self.workflow_run_dir / "workflow_orchestrator_state.json"
        # Ensure all Path objects are converted to strings for JSON
        serializable_state = {}
        for k, v in self.workflow_state.items():
            if isinstance(v, Path): serializable_state[k] = str(v)
            elif isinstance(v, dict): # Simple one-level dict for Path
                serializable_state[k] = {k_inner: str(v_inner) if isinstance(v_inner, Path) else v_inner for k_inner, v_inner in v.items()}
            else: serializable_state[k] = v
        
        with open(state_file, 'w') as f:
            json.dump(serializable_state, f, indent=2)

    # --- Workflow Steps ---
    def step1_verify_all_datasets(self):
        self._start_step("Dataset Verification")
        # Replace placeholder with your actual multi-task dataset verification logic
        all_dataset_verification_stats = verify_all_task_datasets_placeholder(str(self.dataset_root_dir))
        self.workflow_state['dataset_verification_summary'] = all_dataset_verification_stats
        self._complete_step("Dataset Verification")
        return all_dataset_verification_stats

    def step2_run_hyperparameter_search(self):
        self._start_step("Hyperparameter Search (Multi-Task)")
        # HPO script will create its own subdirectories within this path
        hpo_run_output_dir = self.workflow_run_dir / "hpo_search_artifacts"
        hpo_run_output_dir.mkdir(exist_ok=True) # search_v2 will create timestamped dir inside this

        self._log_step_status("Hyperparameter Search", f"Calling HPO script. Output base in: {hpo_run_output_dir}")
        
        # This will include the dataset_root_dir from the orchestrator's init
        fixed_hpo_params_for_run = {
        "dataset_root_dir": str(self.dataset_root_dir) # Use the orchestrator's dataset_root_dir
        # You can add other fixed params here if search_v2.py should not define them
        # e.g., "num_workers": 4 (if you want to set it from orchestrator)
        }

        # Call the main HPO function from your search_v2.py
        # It is expected to return: (best_overall_config_hpo, best_overall_score_hpo, best_overall_trial_best_epoch)
        # This function should handle its own detailed file saving.
        best_hpo_config, best_hpo_score, _ = search_v2.run_experiments_with_single_task_hpo_features(
            hpo_output_base_dir=str(hpo_run_output_dir),
            # Pass any fixed params if search_v2 expects them, e.g.,
            fixed_params_for_hpo_run = {"dataset_root_dir": str(self.dataset_root_dir)}
        )

        if not best_hpo_config:
            raise RuntimeError("Hyperparameter search step did not return a best configuration.")

        # Store key HPO results in workflow state
        self.workflow_state['hpo_summary_results'] = {
            'best_overall_hpo_config_found': best_hpo_config,
            'best_overall_hpo_primary_score_reported': best_hpo_score,
            'hpo_main_artifact_directory': str(hpo_run_output_dir) # Base dir for HPO runs
        }
        self._complete_step("Hyperparameter Search (Multi-Task)")
        return best_hpo_config # Return the best config for the next step

    def step3_train_and_evaluate_final_model(self, best_hpo_config):
        self._start_step("Final Model Training & Test Evaluation")
        final_model_run_output_dir = self.workflow_run_dir / "final_trained_model_artifacts"
        final_model_run_output_dir.mkdir(exist_ok=True)

        self._log_step_status("Final Model Training", f"Using best HPO config. Output in: {final_model_run_output_dir}")
        print(f"   Best HPO Config being used: {json.dumps(best_hpo_config, indent=2)}")
        
        print(f"\n⚠️ CRITICAL WARNING: Test sets will be used for the FIRST and ONLY time! ⚠️")

        final_trainer = FinalMultiTaskModelTrainer(
            dataset_root_dir=str(self.dataset_root_dir),         # CORRECTED name
            best_hyperparams_dict=best_hpo_config,
            final_model_output_dir=str(final_model_run_output_dir) # CORRECTED name
        )
        final_test_metrics = final_trainer.run_final_training_and_evaluation()

        self.workflow_state['final_model_test_metrics_all_tasks'] = final_test_metrics
        self.workflow_state['final_model_main_artifact_directory'] = str(final_model_run_output_dir)
        self._complete_step("Final Model Training & Test Evaluation")
        return final_test_metrics

    def step4_generate_final_workflow_report(self):
        self._start_step("Workflow Report Generation")
        report_file_path = self.workflow_run_dir / "WORKFLOW_COMPLETION_REPORT_MULTITASK.md"
        summary_json_path = self.workflow_run_dir / "workflow_final_summary_multitask.json"
        
        # Gather all data from workflow_state
        ds_summary = self.workflow_state.get('dataset_verification_summary', {})
        hpo_summary = self.workflow_state.get('hpo_summary_results', {})
        final_metrics = self.workflow_state.get('final_model_test_metrics_all_tasks', {})
        
        total_duration_sec = time.time() - self.workflow_state['start_time_unix']

        content = f"# Multi-Task ML Workflow Report\n\n"
        content += f"**Workflow ID:** {self.workflow_state['timestamp']}\n"
        content += f"**Run Directory:** {self.workflow_state['workflow_run_dir']}\n"
        content += f"**Started:** {self.workflow_state['start_time_str']}\n"
        content += f"**Total Duration:** {total_duration_sec/60:.2f} minutes\n\n"

        content += f"## 1. Dataset Summary\n"
        content += f"Dataset Root: {self.workflow_state['dataset_root_dir']}\n"
        content += f"Verification Details:\n```json\n{json.dumps(ds_summary, indent=2)}\n```\n\n"

        content += f"## 2. Hyperparameter Optimization (HPO)\n"
        content += f"HPO Artifacts Location: {hpo_summary.get('hpo_main_artifact_directory', 'N/A')}\n"
        content += f"Best HPO Primary Score (from trial's val set): {hpo_summary.get('best_overall_hpo_primary_score_reported', 'N/A'):.4f}\n"
        content += f"Best HPO Configuration Found:\n```json\n{json.dumps(hpo_summary.get('best_overall_hpo_config_found', {}), indent=2)}\n```\n\n"

        content += f"## 3. Final Model Evaluation (on Test Sets)\n"
        content += f"Final Model Artifacts Location: {self.workflow_state.get('final_model_main_artifact_directory', 'N/A')}\n"
        if final_metrics:
            for task, metrics in final_metrics.items():
                content += f"### Task: {task.upper()}\n"
                content += f"  - F1 Score:  {metrics.get('f1', 0.0):.4f}\n"
                content += f"  - Accuracy:  {metrics.get('accuracy', 0.0):.4f}\n"
                content += f"  - Precision: {metrics.get('precision', 0.0):.4f}\n"
                content += f"  - Recall:    {metrics.get('recall', 0.0):.4f}\n"
                content += f"  - Loss:      {metrics.get('loss', 'N/A')}\n\n" # Loss might be None
        else:
            content += "No final test metrics recorded.\n"

        # Save Report File
        with open(report_file_path, 'w') as f:
            f.write(content)
        self._log_step_status("Workflow Report", f"Report MD saved to {report_file_path}")

        # Save Summary JSON (subset of workflow_state for quick overview)
        summary_data_for_json = {
            'workflow_id': self.workflow_state['timestamp'],
            'total_duration_minutes': total_duration_sec / 60,
            'dataset_root': self.workflow_state['dataset_root_dir'],
            'best_hpo_config': hpo_summary.get('best_overall_hpo_config_found'),
            'best_hpo_score': hpo_summary.get('best_overall_hpo_primary_score_reported'),
            'final_test_metrics': final_metrics,
            'final_model_path': str(Path(self.workflow_state.get('final_model_main_artifact_directory', '')) / "final_trained_model.pt")
        }
        with open(summary_json_path, 'w') as f:
            json.dump(summary_data_for_json, f, indent=2)
        self._log_step_status("Workflow Report", f"Summary JSON saved to {summary_json_path}")

        self._complete_step("Workflow Report Generation")
        return summary_data_for_json


    def run_full_workflow(self):
        self._log_step_status("Full Workflow", "Initiated.")
        try:
            # Step 1
            dataset_summary_info = self.step1_verify_all_datasets() # TODO: Implement properly

            # Step 2
            best_config_from_hpo = self.step2_run_hyperparameter_search()
            if not best_config_from_hpo: # HPO should raise error if it fails, but double check
                print("❌ Critical Error: HPO step did not return a best configuration. Workflow cannot continue.")
                self.workflow_state['error_message'] = "HPO failed to find a best configuration."
                self.workflow_state['failed_at_step'] = self.workflow_state.get('current_step_being_executed', 'HPO')
                self._save_workflow_state()
                return None

            # Step 3
            final_model_results_on_test = self.step3_train_and_evaluate_final_model(best_config_from_hpo)
            
            # Step 4
            overall_workflow_summary = self.step4_generate_final_workflow_report()

            final_duration_sec = time.time() - self.workflow_state['start_time_unix']
            print(f"\n🎉🎉🎉 COMPLETE MULTI-TASK WORKFLOW FINISHED SUCCESSFULLY! 🎉🎉🎉")
            print(f"Total Workflow Duration: {final_duration_sec/60:.2f} minutes.")
            print(f"All artifacts for this workflow run are in: {self.workflow_run_dir}")
            return overall_workflow_summary

        except Exception as e:
            failed_step = self.workflow_state.get('current_step_being_executed', 'Unknown Step')
            self.workflow_state['error_message'] = str(e)
            self.workflow_state['failed_at_step'] = failed_step
            self._save_workflow_state()
            print(f"\n❌❌❌ WORKFLOW EXECUTION FAILED at step: {failed_step} ❌❌❌")
            print(f"Error details: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Orchestrate the Complete Multi-Task ML Workflow")
    parser.add_argument("--dataset_root_dir", type=str, default="/home/avsar/codes-v2/datasets",
                        help="Root directory containing all task-specific dataset CSVs.")
    parser.add_argument("--base_results_dir", type=str, default="./ml_workflow_multitask_orchestrated_runs",
                        help="Base directory to save artifacts for all workflow runs.")
    
    cli_args = parser.parse_args()

    if not Path(cli_args.dataset_root_dir).is_dir():
        print(f"❌ Dataset root directory not found or is not a directory: {cli_args.dataset_root_dir}")
        sys.exit(1)
    
    # Critical: Ensure the HPO script is correctly named and its main function is accessible
    # For example, if your search_v2.py (with HPO logic) is named search_v2_hpo_runner.py:
    # And its main HPO function is run_experiments_with_single_task_hpo_features
    # If search_v2 is not defined due to import issues, this will fail.
    if 'search_v2' not in globals() or not hasattr(search_v2, 'run_experiments_with_single_task_hpo_features'):
        print("ERROR: The HPO script (e.g., search_v2.py) or its main function was not imported correctly.")
        print("Please check the import statements at the top of MultiTaskMLWorkflowOrchestrator.py and ensure")
        print("the HPO script file is named/accessible as expected (e.g., 'search_v2.py' aliased to 'search_v2')")
        print("and contains 'run_experiments_with_single_task_hpo_features'.")
        sys.exit(1)


    orchestrator_instance = MultiTaskMLWorkflowOrchestrator(
        dataset_root_dir_str=cli_args.dataset_root_dir,
        base_results_dir_str=cli_args.base_results_dir
    )
    orchestrator_instance.run_full_workflow()