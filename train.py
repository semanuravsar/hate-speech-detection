import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
import time

import sys
import os

from single_task_bert import SingleTaskBERT
from dataset_loaders import LatentHatredDataset
from utils import CheckpointManager, compute_comprehensive_metrics, print_metrics_summary, save_metrics_to_file


def train_epoch(model, dataloader, optimizer, device, epoch_num=None, verbose=True):
    """Train for one epoch with progress reporting"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        
        # Progress logging
        if verbose and batch_idx > 0 and batch_idx % 50 == 0 and epoch_num is not None:
            avg_loss = total_loss / num_batches
            print(f"    Batch {batch_idx}/{len(dataloader)} - Avg Loss: {avg_loss:.4f}")

    return total_loss / num_batches if num_batches > 0 else 0.0


@torch.no_grad()
def evaluate_model(model, dataloader, device, class_names=None):
    """Comprehensive model evaluation with all metrics"""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Compute comprehensive metrics
    metrics = compute_comprehensive_metrics(all_labels, all_preds, class_names)
    metrics['loss'] = avg_loss
    
    return metrics


def train_model(args, experiment_name=None):
    """
    Main training function with enhanced checkpointing and metrics
    
    Args:
        args: Training arguments
        experiment_name: Name for this experiment (for checkpoint organization)
    """
    print(f"🚀 Starting Training")
    print(f"{'='*60}")
    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print(f"{'='*60}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        base_dir="/home/altemir/Project/scripts/single_task_bert/checkpoints", 
        experiment_name=experiment_name
    )
    
    # Model setup
    model = SingleTaskBERT(dropout=args.dropout).to(device)
    optimizer = AdamW(
        model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Load datasets
    print(f"\n📊 Loading datasets...")
    train_dataset = LatentHatredDataset(args.dataset_path, split="train")
    val_dataset = LatentHatredDataset(args.dataset_path, split="val")
    
    # Get class names for metrics
    class_names = ["not_hate", "implicit_hate", "explicit_hate"]  # Update if you have actual class names
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=getattr(args, 'num_workers', 0)
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        num_workers=getattr(args, 'num_workers', 0)
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_f1 = 0.0
    
    if args.resume and hasattr(args, 'resume_from'):
        try:
            checkpoint = checkpoint_manager.load_checkpoint(
                model, optimizer, checkpoint_type=args.resume_from
            )
            start_epoch = checkpoint.get('epoch', 0)
            best_val_f1 = checkpoint.get('metrics', {}).get('f1', 0.0)
            print(f"📂 Resumed from epoch {start_epoch} with best F1: {best_val_f1:.4f}")
        except Exception as e:
            print(f"⚠️  Could not resume training: {e}")
            print("Starting fresh training...")

    print(f"\n🎯 Training started from epoch {start_epoch + 1}")
    
    # Training history tracking
    training_history = []
    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Training phase
        print("🏋️  Training...")
        train_loss = train_epoch(
            model, train_loader, optimizer, device, 
            epoch_num=epoch+1, verbose=True
        )
        
        # Validation phase
        print("📊 Evaluating...")
        val_metrics = evaluate_model(model, val_loader, device, class_names)
        
        
        # Log epoch results
        epoch_time = time.time() - epoch_start_time
        print(f"\n📈 Epoch {epoch+1} Results (completed in {epoch_time:.1f}s):")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss: {val_metrics['loss']:.4f}")
        print(f"   Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"   Val Precision: {val_metrics['precision']:.4f}")
        print(f"   Val Recall: {val_metrics['recall']:.4f}")
        print(f"   Val F1: {val_metrics['f1']:.4f}")
        
        # Track training history
        history_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'learning_rate': args.lr,
            'epoch_time': epoch_time
        }
        training_history.append(history_entry)
        
        # Check if this is the best model
        is_best = val_metrics['f1'] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics['f1']
            print(f"🎯 NEW BEST MODEL! F1: {best_val_f1:.4f}")
        
        # Save checkpoints
        config_dict = vars(args).copy()
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            metrics=val_metrics,
            config=config_dict,
            is_best=is_best
        )
        
    # Training completed
    total_time = time.time() - start_time
    print(f"\n🎉 Training completed in {total_time/60:.1f} minutes")
    print(f"🏆 Best validation F1: {best_val_f1:.4f}")
    
    # Save training history
    history_path = checkpoint_manager.experiment_dir / "training_history.json"
    import json
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Load and evaluate best model
    print(f"\n📊 Final evaluation with best model...")
    best_checkpoint = checkpoint_manager.load_checkpoint(model, checkpoint_type="best")
    final_val_metrics = evaluate_model(model, val_loader, device, class_names)
    
    print_metrics_summary(final_val_metrics, "Best Model - Validation Performance")
    
    # Save final metrics
    metrics_path = checkpoint_manager.experiment_dir / "final_validation_metrics.json"
    save_metrics_to_file(final_val_metrics, metrics_path)
    
    return {
        'best_val_f1': best_val_f1,
        'final_metrics': final_val_metrics,
        'training_history': training_history,
        'checkpoint_manager': checkpoint_manager
    }


def main(args):
    """Main entry point maintaining backward compatibility"""
    
    # Generate experiment name from configuration
    experiment_name = f"lr{args.lr}_do{args.dropout}_bs{args.batch_size}_ep{args.epochs}"
    
    return train_model(args, experiment_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced BERT training with comprehensive metrics")
    
    # Dataset arguments
    parser.add_argument("--dataset_path", type=str, 
                        default="/home/altemir/Project/scripts/single_task_bert/latent_hatred_3class.csv")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, 
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, 
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, 
                       help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.1, 
                       help="Dropout rate")
    
    
    # Checkpoint arguments
    parser.add_argument("--resume", action="store_true", 
                       help="Resume training from checkpoint")
    parser.add_argument("--resume_from", type=str, default="best", 
                       help="Which checkpoint to resume from (best/epoch_N)")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint.pt", 
                       help="Legacy checkpoint path (for backward compatibility)")
    
    # System arguments
    parser.add_argument("--num_workers", type=int, default=0, 
                       help="Number of data loading workers")
    
    args = parser.parse_args()
    
    try:
        results = main(args)
        print(f"\n✅ Training completed successfully!")
        print(f"🎯 Best F1 Score: {results['best_val_f1']:.4f}")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise