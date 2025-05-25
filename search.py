import argparse
import itertools
import time
import pandas as pd
import torch
import json
from pathlib import Path

import sys
import os

from torch.utils.data import DataLoader

from single_task_bert import SingleTaskBERT
from dataset_loaders import LatentHatredDataset
from utils import CheckpointManager, compute_comprehensive_metrics, print_metrics_summary


class HyperparameterSearchManager:
    """
    Enhanced hyperparameter search with comprehensive tracking and analysis
    """
    
    def __init__(self, dataset_path, results_dir="hyperparameter_search"):
        self.dataset_path = dataset_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Fixed hyperparameters (optimized values)
        self.fixed_dropout = 0.1      # Optimal for BERT fine-tuning
        self.fixed_weight_decay = 0.01  # Standard for transformer models
        self.batch_size = 32
        self.max_epochs = 5  # Train for max epochs
        
        # Initialize datasets
        print("📊 Loading datasets for hyperparameter search...")
        self.train_dataset = LatentHatredDataset(dataset_path, split="train")
        self.val_dataset = LatentHatredDataset(dataset_path, split="val")
        
        self.all_results = []
        self.best_config = None
        self.best_score = 0.0
        
        print(f"💾 Results will be saved to: {self.results_dir}")
        print(f"🔧 Fixed parameters: dropout={self.fixed_dropout}, weight_decay={self.fixed_weight_decay},batch_size={self.batch_size}")
        print(f"🔧 Max epochs per experiment: {self.max_epochs}")
    
    def define_search_space(self):
        """Define the hyperparameter search space (excluding fixed parameters)"""
        return {
            "learning_rate": [1e-5, 2e-5, 3e-5, 5e-5]
        }
    
    def train_single_config(self, config, experiment_id):
        """
        Train model with a specific hyperparameter configuration
        """
        from train import train_model
        
        # Add fixed parameters to config for display
        full_config = config.copy()
        full_config.update({
            "dropout": self.fixed_dropout,
            "weight_decay": self.fixed_weight_decay,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs
        })
        
        print(f"\n🚀 Experiment {experiment_id}")
        print(f"   Configuration: {full_config}")
        print(f"   Training for {self.max_epochs} epochs, best checkpoint will be saved automatically...")
        
        # Create arguments object with fixed parameters
        args = argparse.Namespace(
            dataset_path=self.dataset_path,
            batch_size=self.batch_size,
            epochs=self.max_epochs,
            lr=config["learning_rate"],
            dropout=self.fixed_dropout,        # Fixed value
            weight_decay=self.fixed_weight_decay,  # Fixed value
            resume=False,
            num_workers=0
        )
        
        start_time = time.time()
        
        try:
            # Train the model
            experiment_name = f"search_exp_{experiment_id:03d}"
            results = train_model(args, experiment_name)
            
            training_time = time.time() - start_time
            
            # 🔑 NEW: Extract epoch information from checkpoint manager
            best_epoch = None
            try:
                # The CheckpointManager saves best metrics to "best_metrics.json"
                best_metrics_file = results['checkpoint_manager'].experiment_dir / "best_metrics.json"
                if best_metrics_file.exists():
                    with open(best_metrics_file, 'r') as f:
                        best_metrics_data = json.load(f)
                        best_epoch = best_metrics_data.get('epoch', None)
                        print(f"   🎯 Best epoch: {best_epoch}")
                else:
                    print(f"   ⚠️ Best metrics file not found: {best_metrics_file}")
            except Exception as e:
                print(f"   ⚠️ Could not extract best epoch: {e}")
            
            print(f"   ✅ Training completed in {training_time:.1f}s")
            print(f"   📊 Best F1: {results['best_val_f1']:.4f}")
            
            return {
                'success': True,
                'training_time': training_time,
                'best_val_f1': results['best_val_f1'],
                'best_epoch': best_epoch,  # 🔑 NEW: Include epoch info
                'final_metrics': results['final_metrics'],
                'checkpoint_manager': results['checkpoint_manager']
            }
            
        except Exception as e:
            print(f"   ❌ Training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'training_time': time.time() - start_time,
                'best_epoch': None  # 🔑 NEW: Include None for failed experiments
            }
    
    
    def run_search(self):
        """
        Execute the complete hyperparameter search
        """
        search_space = self.define_search_space()
        
        # Generate all combinations
        param_names = list(search_space.keys())
        param_values = list(search_space.values())
        all_combinations = list(itertools.product(*param_values))
        
        total_experiments = len(all_combinations)
        
        print(f"\n🔬 Starting Hyperparameter Search")
        print(f"{'='*70}")
        print(f"Search space: {search_space}")
        print(f"Fixed parameters: dropout={self.fixed_dropout}, weight_decay={self.fixed_weight_decay},batch_size={self.batch_size}")
        print(f"Max epochs per experiment: {self.max_epochs}")
        print(f"Total experiments: {total_experiments}")
        print(f"{'='*70}")
        
        for experiment_id, param_combination in enumerate(all_combinations, 1):
            config = dict(zip(param_names, param_combination))
            
            print(f"\n{'='*50}")
            print(f"Experiment {experiment_id}/{total_experiments}")
            print(f"{'='*50}")
            
            # Train model with this configuration
            training_results = self.train_single_config(config, experiment_id)
            
            if not training_results['success']:
                # Record failed experiment (include fixed params for completeness)
                experiment_result = {
                    'experiment_id': experiment_id,
                    'config': {**config, 'dropout': self.fixed_dropout, 'weight_decay': self.fixed_weight_decay,'batch_size': self.batch_size},
                    'success': False,
                    'error': training_results['error'],
                    'training_time': training_results['training_time'],
                    'best_epoch': training_results.get('best_epoch')  # 🔑 NEW: Include epoch for failed experiments (will be None)
                }
                self.all_results.append(experiment_result)
                continue
            
            # Evaluate the best model
            final_metrics = training_results['final_metrics']
            
            # Record successful experiment (include fixed params AND epoch info)
            experiment_result = {
                'experiment_id': experiment_id,
                'config': {**config, 'dropout': self.fixed_dropout, 'weight_decay': self.fixed_weight_decay,'batch_size': self.batch_size},
                'success': True,
                'training_time': training_results['training_time'],
                'best_val_f1': training_results['best_val_f1'],
                'best_epoch': training_results.get('best_epoch'),  # 🔑 NEW: Include epoch info
                'final_metrics': {
                    'accuracy': final_metrics['accuracy'],
                    'precision': final_metrics['precision'],
                    'recall': final_metrics['recall'],
                    'f1': final_metrics['f1'],
                    'loss': final_metrics['loss']
                }
            }
            
            self.all_results.append(experiment_result)
            
            # Check if this is the best configuration
            if final_metrics['f1'] > self.best_score:
                self.best_score = final_metrics['f1']
                self.best_config = {**config, 'dropout': self.fixed_dropout, 'weight_decay': self.fixed_weight_decay,'batch_size': self.batch_size}
                self.best_epoch = training_results.get('best_epoch')  # 🔑 NEW: Store best epoch globally
                
                print(f"   🎯 NEW BEST CONFIGURATION!")
                print(f"      F1: {self.best_score:.4f}")
                if self.best_epoch is not None:  # 🔑 NEW: Display epoch if available
                    print(f"      Best Epoch: {self.best_epoch}")
                print(f"      Config: {self.best_config}")
            
            # Save intermediate results
            self.save_results()
        
        # Final analysis
        self.analyze_results()
        
        return self.best_config, self.best_score, self.best_epoch

    
    def save_results(self):
        """Save search results to files"""
        # Save detailed results as JSON
        results_file = self.results_dir / "search_results.json"
        with open(results_file, 'w') as f:
            json.dump(self.all_results, f, indent=2)
        
        # Save summary as CSV
        summary_data = []
        for result in self.all_results:
            if result['success']:
                row = result['config'].copy()
                row.update({
                    'experiment_id': result['experiment_id'],
                    'training_time': result['training_time'],
                    'accuracy': result['final_metrics']['accuracy'],
                    'precision': result['final_metrics']['precision'],
                    'recall': result['final_metrics']['recall'],
                    'f1': result['final_metrics']['f1'],
                    'loss': result['final_metrics']['loss']
                })
                summary_data.append(row)
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            df.to_csv(self.results_dir / "search_summary.csv", index=False)
    
    def analyze_results(self):
        """Analyze and summarize search results"""
        successful_results = [r for r in self.all_results if r['success']]
        
        if not successful_results:
            print("\n❌ No successful experiments to analyze")
            return
        
        print(f"\n📊 HYPERPARAMETER SEARCH ANALYSIS")
        print(f"{'='*70}")
        print(f"Total experiments: {len(self.all_results)}")
        print(f"Successful experiments: {len(successful_results)}")
        print(f"Failed experiments: {len(self.all_results) - len(successful_results)}")
        print(f"Fixed parameters: dropout={self.fixed_dropout}, weight_decay={self.fixed_weight_decay}")
        
        # Best configuration
        print(f"\n🏆 BEST CONFIGURATION:")
        print(f"   F1 Score: {self.best_score:.4f}")
        print(f"   Configuration:")
        for param, value in self.best_config.items():
            print(f"     {param}: {value}")
        
        # Top 5 configurations
        sorted_results = sorted(successful_results, 
                               key=lambda x: x['final_metrics']['f1'], 
                               reverse=True)
        
        print(f"\n📈 TOP 5 CONFIGURATIONS:")
        for i, result in enumerate(sorted_results[:5], 1):
            metrics = result['final_metrics']
            print(f"   {i}. F1={metrics['f1']:.4f}, Acc={metrics['accuracy']:.4f}, "
                  f"Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}")
            print(f"      Config: {result['config']}")
        
        # Parameter analysis (only for variable parameters)
        self.analyze_parameter_importance(successful_results)
        
        # Save analysis
        analysis = {
            'best_config': self.best_config,
            'best_score': self.best_score,
            'fixed_parameters': {
                'dropout': self.fixed_dropout,
                'weight_decay': self.fixed_weight_decay,
                'batch_size': self.batch_size
            },
            'top_5_configs': sorted_results[:5],
            'total_experiments': len(self.all_results),
            'successful_experiments': len(successful_results)
        }
        
        with open(self.results_dir / "analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n💾 Analysis saved to: {self.results_dir}")
    
    def analyze_parameter_importance(self, results):
        """Analyze which parameters have the most impact on performance"""
        print(f"\n🔍 PARAMETER IMPACT ANALYSIS:")
        
        df = pd.DataFrame([
            {**r['config'], 'f1': r['final_metrics']['f1']} 
            for r in results
        ])
        
        # Only analyze variable parameters (not fixed ones)
        variable_params = ['learning_rate', 'epochs']
        
        for param in variable_params:
            if param in df.columns:
                param_analysis = df.groupby(param)['f1'].agg(['mean', 'std', 'count'])
                print(f"\n   {param.upper()}:")
                for value, stats in param_analysis.iterrows():
                    print(f"     {value}: F1={stats['mean']:.4f}±{stats['std']:.4f} (n={stats['count']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter search for BERT model")
    parser.add_argument("--dataset_path", type=str, 
                       default="/home/altemir/hate-speech-detection/datasets/latent_hatred_3class.csv",
                       help="Path to the dataset CSV file")
    parser.add_argument("--results_dir", type=str, default="hyperparameter_search",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    try:
        search_manager = HyperparameterSearchManager(args.dataset_path, args.results_dir)
        best_config, best_score = search_manager.run_search()
        
        print(f"\n✅ Hyperparameter search completed successfully!")
        print(f"🎯 Best configuration will be used for final model training")
        
    except Exception as e:
        print(f"\n❌ Hyperparameter search failed: {e}")
        raise