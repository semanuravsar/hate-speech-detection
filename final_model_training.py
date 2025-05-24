"""
Final Model Training Script

This script trains the best configuration found during hyperparameter search
on the combined train+validation set, then evaluates on the held-out test set.

This is the FINAL step before deployment - test set is used only once!
"""

import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import time
from pathlib import Path

import sys
import os

from single_task_bert import SingleTaskBERT
from dataset_loaders import LatentHatredDataset
from utils import CheckpointManager, compute_comprehensive_metrics, print_metrics_summary, save_metrics_to_file


class FinalModelTrainer:
    """
    Train final model using best hyperparameters on full training data
    """
    
    def __init__(self, dataset_path, best_config, results_dir="final_model"):
        self.dataset_path = dataset_path
        self.best_config = best_config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        print(f"🎯 Final Model Training")
        print(f"{'='*60}")
        print(f"Dataset: {dataset_path}")
        print(f"Results directory: {self.results_dir}")
        print(f"Best configuration: {best_config}")
        print(f"{'='*60}")
    
    def load_datasets(self):
        """
        Load train, validation, and test datasets
        For final training: combine train+val, evaluate on test
        """
        print(f"📊 Loading datasets...")
        
        # Load individual splits
        train_dataset = LatentHatredDataset(self.dataset_path, split="train")
        val_dataset = LatentHatredDataset(self.dataset_path, split="val")
        test_dataset = LatentHatredDataset(self.dataset_path, split="test")
        
        # Combine train and validation for final training
        combined_train_dataset = ConcatDataset([train_dataset, val_dataset])
        
        print(f"📈 Dataset sizes:")
        print(f"   Original train: {len(train_dataset):,} samples")
        print(f"   Original val: {len(val_dataset):,} samples")
        print(f"   Combined train: {len(combined_train_dataset):,} samples")
        print(f"   Test (hold-out): {len(test_dataset):,} samples")
        
        return combined_train_dataset, test_dataset
    
    def train_final_model(self):
        """
        Train the final model using best hyperparameters on combined train+val data
        """
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Using device: {device}")
        
        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(
            base_dir=str(self.results_dir), 
            experiment_name="final_model"
        )
        
        # Load datasets
        combined_train_dataset, test_dataset = self.load_datasets()
        
        # Create data loaders
        train_loader = DataLoader(
            combined_train_dataset,
            batch_size=self.best_config["batch_size"],
            shuffle=True,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.best_config["batch_size"],
            shuffle=False,
            num_workers=0
        )
        
        # Initialize model with best hyperparameters
        model = SingleTaskBERT(dropout=self.best_config["dropout"]).to(device)
        optimizer = AdamW(
            model.parameters(),
            lr=self.best_config["learning_rate"],
            weight_decay=self.best_config.get("weight_decay", 0.01)
        )
        
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-7
        )
        
        print(f"\n🚀 Starting final model training...")
        print(f"   Epochs: {self.best_config['epochs']}")
        print(f"   Learning rate: {self.best_config['learning_rate']}")
        print(f"   Dropout: {self.best_config['dropout']}")
        print(f"   Batch size: {self.best_config['batch_size']}")
        
        # Training loop
        training_history = []
        start_time = time.time()
        
        for epoch in range(self.best_config["epochs"]):
            epoch_start_time = time.time()
            
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.best_config['epochs']}")
            print(f"{'='*50}")
            
            # Training phase
            model.train()
            total_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Progress logging
                if batch_idx > 0 and batch_idx % 100 == 0:
                    avg_loss = total_loss / num_batches
                    print(f"    Batch {batch_idx}/{len(train_loader)} - Avg Loss: {avg_loss:.4f}")
            
            avg_train_loss = total_loss / num_batches
            scheduler.step(avg_train_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            epoch_time = time.time() - epoch_start_time
            
            print(f"\n📊 Epoch {epoch+1} completed in {epoch_time:.1f}s:")
            print(f"   Train Loss: {avg_train_loss:.4f}")
            print(f"   Learning Rate: {current_lr:.2e}")
            
            # Save training history
            history_entry = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                'learning_rate': current_lr,
                'epoch_time': epoch_time
            }
            training_history.append(history_entry)
            
            # Save checkpoint
            checkpoint_manager.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                metrics={'train_loss': avg_train_loss},
                config=self.best_config,
                is_best=False,  # We'll determine best after final evaluation
                save_periodic=(epoch + 1) % 5 == 0
            )
        
        total_training_time = time.time() - start_time
        print(f"\n🎉 Final model training completed in {total_training_time/60:.1f} minutes")
        
        # Save training history
        history_path = self.results_dir / "final_model" / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        return model, test_loader, checkpoint_manager
    
    def evaluate_on_test_set(self, model, test_loader, checkpoint_manager):
        """
        CRITICAL: Evaluate final model on test set - ONLY DONE ONCE!
        """
        device = next(model.parameters()).device
        
        print(f"\n🔬 FINAL EVALUATION ON TEST SET")
        print(f"{'='*60}")
        print(f"⚠️  This is the FINAL evaluation - test set used for first time!")
        print(f"⚠️  No more hyperparameter tuning allowed after this!")
        print(f"{'='*60}")
        
        # Evaluate model
        from train import evaluate_model
        class_names = ["not_hate", "implicit_hate", "explicit_hate"]
        
        start_time = time.time()
        test_metrics = evaluate_model(model, test_loader, device, class_names)
        eval_time = time.time() - start_time
        
        print(f"\n📊 TEST SET EVALUATION RESULTS:")
        print_metrics_summary(test_metrics, "FINAL MODEL - TEST SET PERFORMANCE")
        
        # Save comprehensive test results
        test_results = {
            'test_metrics': test_metrics,
            'evaluation_time': eval_time,
            'model_config': self.best_config,
            'dataset_path': str(self.dataset_path),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save to multiple formats for easy access
        results_path = self.results_dir / "final_test_results.json"
        save_metrics_to_file(test_results, results_path)
        
        # Save summary metrics to CSV
        summary_df = pd.DataFrame([{
            'accuracy': test_metrics['accuracy'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'f1': test_metrics['f1'],
            'loss': test_metrics['loss']
        }])
        summary_df.to_csv(self.results_dir / "final_test_summary.csv", index=False)
        
        # Update checkpoint as final best model
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=None,
            epoch=self.best_config["epochs"],
            metrics=test_metrics,
            config=self.best_config,
            is_best=True,
            save_periodic=False
        )
        
        return test_metrics
    
    def generate_final_report(self, test_metrics):
        """
        Generate comprehensive final report
        """
        report_path = self.results_dir / "FINAL_MODEL_REPORT.md"
        
        report_content = f"""# Final Model Training Report

## Configuration
- **Dataset**: {self.dataset_path}
- **Training Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Best Hyperparameters**: {json.dumps(self.best_config, indent=2)}

## Training Details
- **Training Data**: Combined train + validation sets
- **Test Data**: Held-out test set (used only once for final evaluation)
- **Model**: BERT-base-uncased with classification head
- **Total Parameters**: ~110M (only classification head trained)

## Final Test Set Performance

### Overall Metrics
- **Accuracy**: {test_metrics['accuracy']:.4f}
- **Precision**: {test_metrics['precision']:.4f}
- **Recall**: {test_metrics['recall']:.4f}
- **F1 Score**: {test_metrics['f1']:.4f}
- **Loss**: {test_metrics['loss']:.4f}

### Per-Class Performance
"""
        
        for i, class_name in enumerate(test_metrics['class_names']):
            report_content += f"""
#### {class_name}
- **Precision**: {test_metrics['precision_per_class'][i]:.4f}
- **Recall**: {test_metrics['recall_per_class'][i]:.4f}
- **F1 Score**: {test_metrics['f1_per_class'][i]:.4f}
"""
        
        report_content += f"""
### Confusion Matrix
```
{test_metrics['confusion_matrix']}
```

## Model Deployment Checklist
- [ ] Model performance meets requirements
- [ ] Model saved and ready for deployment
- [ ] Documentation completed
- [ ] Testing pipeline verified

## Files Generated
- `final_model/checkpoint_best.pt` - Best model weights
- `final_test_results.json` - Detailed test results
- `final_test_summary.csv` - Summary metrics
- `training_history.json` - Training progress

**⚠️ IMPORTANT**: This test set evaluation was performed only once. 
No further hyperparameter tuning should be done based on these results.
"""
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"\n📄 Final report generated: {report_path}")
        
    def run_final_training(self):
        """
        Execute the complete final training pipeline
        """
        print(f"🎯 Starting Final Model Training Pipeline")
        
        # Step 1: Train final model
        model, test_loader, checkpoint_manager = self.train_final_model()
        
        # Step 2: Evaluate on test set (ONLY ONCE!)
        test_metrics = self.evaluate_on_test_set(model, test_loader, checkpoint_manager)
        
        # Step 3: Generate final report
        self.generate_final_report(test_metrics)
        
        print(f"\n🎉 FINAL MODEL TRAINING COMPLETED!")
        print(f"🏆 Final Test F1 Score: {test_metrics['f1']:.4f}")
        print(f"📁 All results saved to: {self.results_dir}")
        print(f"📄 See FINAL_MODEL_REPORT.md for complete summary")
        
        return test_metrics


def load_best_config_from_search(search_results_path):
    """
    Load the best configuration from hyperparameter search results
    """
    if not os.path.exists(search_results_path):
        raise FileNotFoundError(f"Search results not found: {search_results_path}")
    
    with open(search_results_path, 'r') as f:
        analysis = json.load(f)
    
    best_config = analysis['best_config']
    best_score = analysis['best_score']
    
    print(f"📂 Loaded best configuration from search:")
    print(f"   F1 Score: {best_score:.4f}")
    print(f"   Config: {best_config}")
    
    return best_config


def main():
    """
    Main function for final model training
    """
    parser = argparse.ArgumentParser(
        description="Train final model with best hyperparameters on full dataset"
    )
    
    parser.add_argument("--dataset_path", type=str, default="/home/altemir/hate-speech-detection/datasets/latent_hatred_3class.csv",
                       help="Path to the dataset CSV file")
    parser.add_argument("--search_results", type=str,
                       default="hyperparameter_search/analysis.json",
                       help="Path to hyperparameter search results")
    parser.add_argument("--config_file", type=str,
                       help="Path to JSON file with best configuration (alternative to search_results)")
    parser.add_argument("--results_dir", type=str, default="final_model",
                       help="Directory to save final model results")
    
    # Allow manual specification of best config
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--dropout", type=float, help="Dropout rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--weight_decay", type=float, help="Weight decay")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    
    args = parser.parse_args()
    
    # Determine best configuration
    if args.config_file:
        # Load from specified config file
        with open(args.config_file, 'r') as f:
            best_config = json.load(f)
    elif all(getattr(args, param) is not None for param in ['lr', 'dropout', 'batch_size', 'epochs']):
        # Use manually specified parameters
        best_config = {
            'learning_rate': args.lr,
            'dropout': args.dropout,
            'batch_size': args.batch_size,
            'weight_decay': args.weight_decay or 0.01,
            'epochs': args.epochs
        }
        print(f"🔧 Using manually specified configuration: {best_config}")
    else:
        # Load from hyperparameter search results
        best_config = load_best_config_from_search(args.search_results)
    
    # Verify dataset exists
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
    
    # Run final training
    try:
        trainer = FinalModelTrainer(args.dataset_path, best_config, args.results_dir)
        final_metrics = trainer.run_final_training()
        
        print(f"\n✅ SUCCESS: Final model training completed!")
        print(f"🎯 Ready for deployment with F1 score: {final_metrics['f1']:.4f}")
        
        return final_metrics
        
    except Exception as e:
        print(f"\n❌ Final model training failed: {e}")
        raise


if __name__ == "__main__":
    import pandas as pd  # Add missing import
    main()