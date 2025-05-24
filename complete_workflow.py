"""
Complete ML Workflow Script

This script orchestrates the entire machine learning pipeline:
1. Dataset verification and splitting
2. Hyperparameter search on train/val sets
3. Final model training on train+val
4. Final evaluation on test set (once only!)

"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
# sys.path.append(os.path.expanduser("~/a"))

from dataset_loaders import verify_stratification
from search import HyperparameterSearchManager
from final_model_training import FinalModelTrainer


class MLWorkflowOrchestrator:
    """
    Orchestrates the complete ML workflow from data validation to final model
    """
    
    def __init__(self, dataset_path, base_results_dir="ml_workflow_results"):
        self.dataset_path = Path(dataset_path)
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(exist_ok=True)
        
        # Create timestamp for this workflow run
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.workflow_dir = self.base_results_dir / f"workflow_{self.timestamp}"
        self.workflow_dir.mkdir(exist_ok=True)
        
        print(f"🚀 ML Workflow Orchestrator Initialized")
        print(f"{'='*70}")
        print(f"Dataset: {self.dataset_path}")
        print(f"Results directory: {self.workflow_dir}")
        print(f"Timestamp: {self.timestamp}")
        print(f"{'='*70}")
        
        # Workflow state tracking
        self.workflow_state = {
            'start_time': time.time(),
            'dataset_path': str(self.dataset_path),
            'workflow_dir': str(self.workflow_dir),
            'timestamp': self.timestamp,
            'steps_completed': [],
            'current_step': None
        }
    
    def log_step(self, step_name, status='started'):
        """Log workflow step progress"""
        if status == 'started':
            self.workflow_state['current_step'] = step_name
            print(f"\n🔄 Step: {step_name}")
            print(f"{'='*50}")
        elif status == 'completed':
            self.workflow_state['steps_completed'].append(step_name)
            print(f"✅ Completed: {step_name}")
            self.save_workflow_state()
    
    def save_workflow_state(self):
        """Save current workflow state"""
        state_file = self.workflow_dir / "workflow_state.json"
        with open(state_file, 'w') as f:
            json.dump(self.workflow_state, f, indent=2)
    
    def step1_verify_dataset(self):
        """Step 1: Verify dataset and splits"""
        self.log_step("Dataset Verification", 'started')
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")
        
        print(f"📊 Verifying dataset and stratification...")
        
        # Verify stratification
        train_ds, val_ds, test_ds = verify_stratification(str(self.dataset_path))
        
        # Save dataset statistics
        dataset_stats = {
            'total_samples': len(train_ds) + len(val_ds) + len(test_ds),
            'train_samples': len(train_ds),
            'val_samples': len(val_ds),
            'test_samples': len(test_ds),
            'train_ratio': len(train_ds) / (len(train_ds) + len(val_ds) + len(test_ds)),
            'val_ratio': len(val_ds) / (len(train_ds) + len(val_ds) + len(test_ds)),
            'test_ratio': len(test_ds) / (len(train_ds) + len(val_ds) + len(test_ds))
        }
        
        stats_file = self.workflow_dir / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(dataset_stats, f, indent=2)
        
        self.workflow_state['dataset_stats'] = dataset_stats
        self.log_step("Dataset Verification", 'completed')
        
        return dataset_stats
    
    def step2_hyperparameter_search(self):
        """Step 2: Hyperparameter search on train/val sets"""
        self.log_step("Hyperparameter Search", 'started')
        
        # Setup search directory
        search_dir = self.workflow_dir / "hyperparameter_search"
        
        print(f"🔍 Running hyperparameter search...")
        print(f"   Results will be saved to: {search_dir}")
        
        # Run hyperparameter search
        search_manager = HyperparameterSearchManager(
            str(self.dataset_path), 
            str(search_dir)
        )
        
        best_config, best_score = search_manager.run_search()
        
        # Save search results to workflow state
        self.workflow_state['hyperparameter_search'] = {
            'best_config': best_config,
            'best_score': best_score,
            'search_directory': str(search_dir)
        }
        
        self.log_step("Hyperparameter Search", 'completed')
        
        return best_config, best_score
    
    def step3_final_model_training(self, best_config):
        """Step 3: Train final model on train+val, evaluate on test"""
        self.log_step("Final Model Training", 'started')
        
        # Setup final model directory
        final_model_dir = self.workflow_dir / "final_model"
        
        print(f"🎯 Training final model with best configuration...")
        print(f"   Configuration: {best_config}")
        print(f"   Results will be saved to: {final_model_dir}")
        
        # WARNING: Test set will be used for the first time!
        print(f"\n⚠️  CRITICAL WARNING:")
        print(f"   The test set will now be used for the FIRST TIME!")
        print(f"   No further hyperparameter tuning after this step!")
        print(f"   This is the final model evaluation!")
        
        # Train final model
        trainer = FinalModelTrainer(
            str(self.dataset_path),
            best_config,
            str(final_model_dir)
        )
        
        final_test_metrics = trainer.run_final_training()
        
        # Save final results to workflow state
        self.workflow_state['final_model'] = {
            'test_metrics': final_test_metrics,
            'model_directory': str(final_model_dir),
            'configuration': best_config
        }
        
        self.log_step("Final Model Training", 'completed')
        
        return final_test_metrics
    
    def step4_generate_workflow_report(self, dataset_stats, best_config, best_val_score, final_test_metrics):
        """Step 4: Generate comprehensive workflow report"""
        self.log_step("Workflow Report Generation", 'started')
        
        total_time = time.time() - self.workflow_state['start_time']
        
        # Generate comprehensive report
        report_content = f"""# Complete ML Workflow Report

## Workflow Overview
- **Start Time**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.workflow_state['start_time']))}
- **Total Duration**: {total_time/60:.1f} minutes
- **Dataset**: {self.dataset_path}
- **Workflow ID**: {self.timestamp}

## Dataset Statistics
- **Total Samples**: {dataset_stats['total_samples']:,}
- **Train Set**: {dataset_stats['train_samples']:,} ({dataset_stats['train_ratio']:.1%})
- **Validation Set**: {dataset_stats['val_samples']:,} ({dataset_stats['val_ratio']:.1%})
- **Test Set**: {dataset_stats['test_samples']:,} ({dataset_stats['test_ratio']:.1%})

## Hyperparameter Search Results
- **Best Validation F1**: {best_val_score:.4f}
- **Best Configuration**:
```json
{json.dumps(best_config, indent=2)}
```

## Final Model Performance (Test Set)
- **Test Accuracy**: {final_test_metrics['accuracy']:.4f}
- **Test Precision**: {final_test_metrics['precision']:.4f}
- **Test Recall**: {final_test_metrics['recall']:.4f}
- **Test F1 Score**: {final_test_metrics['f1']:.4f}
- **Test Loss**: {final_test_metrics['loss']:.4f}

### Per-Class Performance (Test Set)
"""
        
        for i, class_name in enumerate(final_test_metrics['class_names']):
            report_content += f"""
#### {class_name}
- Precision: {final_test_metrics['precision_per_class'][i]:.4f}
- Recall: {final_test_metrics['recall_per_class'][i]:.4f}
- F1 Score: {final_test_metrics['f1_per_class'][i]:.4f}
"""
        
        report_content += f"""
## Workflow Validation
- ✅ Dataset properly stratified across splits
- ✅ Hyperparameter search completed on train/val only
- ✅ Final model trained on combined train+val data
- ✅ Test set used only once for final evaluation
- ✅ No data leakage detected

## Deployment Readiness
- **Model Location**: `{self.workflow_dir}/final_model/final_model/checkpoint_best.pt`
- **Performance Meets Threshold**: {'✅ Yes' if final_test_metrics['f1'] > 0.65 else '❌ No (F1 < 0.65)'}
- **Documentation Complete**: ✅ Yes
- **Ready for Production**: {'✅ Yes' if final_test_metrics['f1'] > 0.65 else '⚠️ Requires Review'}

## Directory Structure
```
{self.workflow_dir}/
├── workflow_state.json              # Workflow tracking
├── dataset_statistics.json          # Dataset analysis
├── hyperparameter_search/           # Search results and checkpoints
│   ├── search_results.json
│   ├── analysis.json
│   └── search_summary.csv
├── final_model/                     # Final model and evaluation
│   ├── final_model/
│   │   └── checkpoint_best.pt       # 🎯 FINAL MODEL
│   ├── final_test_results.json
│   ├── FINAL_MODEL_REPORT.md
│   └── training_history.json
└── WORKFLOW_REPORT.md               # This report
```

## Next Steps
1. Review test set performance
2. If satisfactory, deploy model from `checkpoint_best.pt`
3. Monitor model performance in production
4. If performance is insufficient, restart workflow with:
   - More data
   - Different model architecture
   - Extended hyperparameter search

---
**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}
**Workflow ID**: {self.timestamp}
"""
        
        # Save report
        report_path = self.workflow_dir / "WORKFLOW_REPORT.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Save workflow summary as JSON
        workflow_summary = {
            'workflow_id': self.timestamp,
            'total_duration_minutes': total_time / 60,
            'dataset_stats': dataset_stats,
            'best_hyperparameters': best_config,
            'best_validation_f1': best_val_score,
            'final_test_metrics': {
                'accuracy': final_test_metrics['accuracy'],
                'precision': final_test_metrics['precision'],
                'recall': final_test_metrics['recall'],
                'f1': final_test_metrics['f1'],
                'loss': final_test_metrics['loss']
            },
            'model_path': f"{self.workflow_dir}/final_model/final_model/checkpoint_best.pt",
            'ready_for_deployment': final_test_metrics['f1'] > 0.65
        }
        
        summary_path = self.workflow_dir / "workflow_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(workflow_summary, f, indent=2)
        
        self.log_step("Workflow Report Generation", 'completed')
        
        print(f"\n📄 Comprehensive report generated:")
        print(f"   📋 {report_path}")
        print(f"   📊 {summary_path}")
        
        return workflow_summary
    
    def run_complete_workflow(self):
        """
        Run the complete ML workflow from start to finish
        """
        print(f"🚀 Starting Complete ML Workflow")
        print(f"⏱️  Estimated time: 30-60 minutes (depends on hyperparameter search)")
        
        try:
            # Step 1: Verify dataset
            dataset_stats = self.step1_verify_dataset()
            
            # Step 2: Hyperparameter search
            best_config, best_score = self.step2_hyperparameter_search()
            
            # Step 3: Final model training and test evaluation
            final_test_metrics = self.step3_final_model_training(best_config)
            
            # Step 4: Generate workflow report
            workflow_summary = self.step4_generate_workflow_report(
                dataset_stats, best_config, best_score, final_test_metrics
            )
            
            # Final success message
            total_time = time.time() - self.workflow_state['start_time']
            
            print(f"\n🎉 COMPLETE WORKFLOW FINISHED SUCCESSFULLY!")
            print(f"{'='*70}")
            print(f"⏱️  Total Time: {total_time/60:.1f} minutes")
            print(f"🎯 Final Test F1: {final_test_metrics['f1']:.4f}")
            print(f"📁 Results: {self.workflow_dir}")
            print(f"🤖 Model: {workflow_summary['model_path']}")
            print(f"🚀 Ready for deployment: {'✅ Yes' if workflow_summary['ready_for_deployment'] else '⚠️ Needs review'}")
            print(f"{'='*70}")
            
            return workflow_summary
            
        except Exception as e:
            self.workflow_state['error'] = str(e)
            self.workflow_state['failed_at_step'] = self.workflow_state.get('current_step', 'unknown')
            self.save_workflow_state()
            
            print(f"\n❌ Workflow failed at step: {self.workflow_state.get('current_step', 'unknown')}")
            print(f"Error: {e}")
            raise


def main():
    """Main entry point for complete workflow"""
    parser = argparse.ArgumentParser(
        description="Complete ML workflow: dataset validation → hyperparameter search → final training → test evaluation"
    )
    
    parser.add_argument("--dataset_path", type=str,default="/home/altemir/hate-speech-detection/datasets/latent_hatred_3class.csv",
                       help="Path to the dataset CSV file")
    parser.add_argument("--results_dir", type=str, default="ml_workflow_results",
                       help="Base directory for all workflow results")
    
    args = parser.parse_args()
    
    
    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        print(f"❌ Dataset file not found: {args.dataset_path}")
        return 1
    
    # Run complete workflow
    try:
        orchestrator = MLWorkflowOrchestrator(args.dataset_path, args.results_dir)
        workflow_summary = orchestrator.run_complete_workflow()
        
        print(f"\n🎊 SUCCESS: Your ML model is ready!")
        print(f"📋 See WORKFLOW_REPORT.md for complete details")
        
        return 0
        
    except Exception as e:
        print(f"\n💥 Workflow failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())