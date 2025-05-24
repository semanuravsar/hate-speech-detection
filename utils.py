import re
from transformers import BertTokenizer
import torch
import os
import glob
import json
import shutil
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def preprocess_text(text):
    """Enhanced text preprocessing with better handling"""
    text = str(text).lower()
    text = re.sub(r"http\S+", "<url>", text)
    text = re.sub(r"@\w+", "<user>", text)
    text = re.sub(r"#\w+", "<hashtag>", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII
    text = re.sub(r"\s+", " ", text).strip()
    return text


def encode_text(text, max_len=128):
    """Enhanced text encoding with error handling"""
    try:
        return tokenizer(
            text, 
            truncation=True, 
            padding="max_length", 
            max_length=max_len, 
            return_tensors="pt"
        )
    except Exception as e:
        print(f"Warning: Error encoding text '{text[:50]}...': {e}")
        # Return empty encoding as fallback
        return tokenizer(
            "", 
            truncation=True, 
            padding="max_length", 
            max_length=max_len, 
            return_tensors="pt"
        )


class CheckpointManager:
    """
    Advanced checkpoint management system that prevents overwriting and tracks best models
    """
    
    def __init__(self, base_dir="checkpoints", experiment_name=None):
        """
        Args:
            base_dir: Base directory for all checkpoints
            experiment_name: Name of current experiment (auto-generated if None)
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"exp_{timestamp}"
        
        self.experiment_dir = self.base_dir / experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        self.best_metrics = {}
        self.best_checkpoint_path = None
        
        print(f"📁 Checkpoint manager initialized: {self.experiment_dir}")
    
    def save_checkpoint(self, model, optimizer, epoch, metrics=None, config=None, 
                       is_best=False, save_periodic=True):
        """
        Save checkpoint with comprehensive metadata
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            epoch: Current epoch
            metrics: Dictionary of metrics
            config: Training configuration
            is_best: Whether this is the best model so far
            save_periodic: Whether to save periodic checkpoints
        """
        timestamp = datetime.now().isoformat()
        
        checkpoint_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "timestamp": timestamp,
            "metrics": metrics or {},
            "config": config or {}
        }
        
        # Always save latest checkpoint
        latest_path = self.experiment_dir / "checkpoint_latest.pt"
        torch.save(checkpoint_data, latest_path)
        
        # Save periodic checkpoint (doesn't overwrite)
        if save_periodic and epoch % 5 == 0:
            periodic_path = self.experiment_dir / f"checkpoint_epoch_{epoch:03d}.pt"
            torch.save(checkpoint_data, periodic_path)
            print(f"💾 Periodic checkpoint saved: {periodic_path.name}")
        
        # Save best checkpoint
        if is_best and metrics:
            best_path = self.experiment_dir / "checkpoint_best.pt"
            torch.save(checkpoint_data, best_path)
            self.best_checkpoint_path = best_path
            self.best_metrics = metrics.copy()
            print(f"🏆 Best checkpoint saved: {best_path.name}")
            
            # Save best metrics separately for easy access
            metrics_path = self.experiment_dir / "best_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump({
                    "epoch": epoch,
                    "metrics": metrics,
                    "timestamp": timestamp
                }, f, indent=2)
        
        return latest_path
    
    def load_checkpoint(self, model, optimizer=None, checkpoint_type="best"):
        """
        Load checkpoint with enhanced error handling
        
        Args:
            model: PyTorch model
            optimizer: Optimizer (optional)
            checkpoint_type: "best", "latest", or specific epoch number
        """
        if checkpoint_type == "best":
            checkpoint_path = self.experiment_dir / "checkpoint_best.pt"
        elif checkpoint_type == "latest":
            checkpoint_path = self.experiment_dir / "checkpoint_latest.pt"
        elif isinstance(checkpoint_type, int):
            checkpoint_path = self.experiment_dir / f"checkpoint_epoch_{checkpoint_type:03d}.pt"
        else:
            checkpoint_path = Path(checkpoint_type)  # Direct path
        
        if not checkpoint_path.exists():
            available = self.list_available_checkpoints()
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                f"Available checkpoints: {[c['name'] for c in available]}"
            )
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint["model_state_dict"])
            
            if optimizer and "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            epoch = checkpoint.get("epoch", 0)
            metrics = checkpoint.get("metrics", {})
            
            print(f"✅ Loaded checkpoint: {checkpoint_path.name} (epoch {epoch})")
            if metrics:
                print(f"   Metrics: {metrics}")
            
            return checkpoint
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint {checkpoint_path}: {e}")
    
    def list_available_checkpoints(self):
        """List all available checkpoints for this experiment"""
        checkpoints = []
        
        # Best checkpoint
        best_path = self.experiment_dir / "checkpoint_best.pt"
        if best_path.exists():
            try:
                ckpt = torch.load(best_path, map_location='cpu')
                checkpoints.append({
                    'name': 'best',
                    'path': best_path,
                    'epoch': ckpt.get('epoch', 0),
                    'metrics': ckpt.get('metrics', {}),
                    'timestamp': ckpt.get('timestamp', 'unknown')
                })
            except:
                pass
        
        # Latest checkpoint
        latest_path = self.experiment_dir / "checkpoint_latest.pt"
        if latest_path.exists():
            try:
                ckpt = torch.load(latest_path, map_location='cpu')
                checkpoints.append({
                    'name': 'latest',
                    'path': latest_path,
                    'epoch': ckpt.get('epoch', 0),
                    'metrics': ckpt.get('metrics', {}),
                    'timestamp': ckpt.get('timestamp', 'unknown')
                })
            except:
                pass
        
        # Periodic checkpoints
        for periodic_path in sorted(self.experiment_dir.glob("checkpoint_epoch_*.pt")):
            try:
                ckpt = torch.load(periodic_path, map_location='cpu')
                epoch_num = int(periodic_path.stem.split('_')[-1])
                checkpoints.append({
                    'name': f'epoch_{epoch_num}',
                    'path': periodic_path,
                    'epoch': ckpt.get('epoch', epoch_num),
                    'metrics': ckpt.get('metrics', {}),
                    'timestamp': ckpt.get('timestamp', 'unknown')
                })
            except:
                pass
        
        return sorted(checkpoints, key=lambda x: x['epoch'])
    
    def cleanup_old_checkpoints(self, keep_last_n=3):
        """Remove old periodic checkpoints to save disk space"""
        periodic_files = list(self.experiment_dir.glob("checkpoint_epoch_*.pt"))
        
        if len(periodic_files) <= keep_last_n:
            return
        
        # Sort by epoch number
        def extract_epoch(path):
            try:
                return int(path.stem.split('_')[-1])
            except:
                return 0
        
        periodic_files.sort(key=extract_epoch)
        
        # Remove old checkpoints
        files_to_remove = periodic_files[:-keep_last_n]
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                print(f"🗑️  Removed old checkpoint: {file_path.name}")
            except OSError as e:
                print(f"Warning: Could not remove {file_path.name}: {e}")


def compute_comprehensive_metrics(y_true, y_pred, class_names=None, average='macro'):
    """
    Compute comprehensive classification metrics including precision, recall, F1
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        class_names: Names of classes (optional)
        average: Averaging strategy for multi-class metrics
    
    Returns:
        Dictionary with all metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average, zero_division=0)
    recall = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Classification report
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(precision_per_class))]
    
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                 output_dict=True, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(), 
        'f1_per_class': f1_per_class.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'class_names': class_names
    }
    
    return metrics


def print_metrics_summary(metrics, title="Model Performance"):
    """Print a nicely formatted summary of metrics"""
    print(f"\n📊 {title}")
    print("=" * 60)
    
    # Overall metrics
    print(f"Overall Performance:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    # Per-class metrics
    print(f"\nPer-Class Performance:")
    class_names = metrics.get('class_names', [f"Class_{i}" for i in range(len(metrics['f1_per_class']))])
    
    for i, class_name in enumerate(class_names):
        print(f"  {class_name}:")
        print(f"    Precision: {metrics['precision_per_class'][i]:.4f}")
        print(f"    Recall:    {metrics['recall_per_class'][i]:.4f}")
        print(f"    F1:        {metrics['f1_per_class'][i]:.4f}")
    
    # Confusion Matrix
    print(f"\nConfusion Matrix:")
    cm = metrics['confusion_matrix']
    print("    " + "  ".join([f"{name:>8}" for name in class_names]))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:>8}" + "".join([f"{val:>10}" for val in row]))


def save_metrics_to_file(metrics, filepath):
    """Save metrics to JSON file"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"💾 Metrics saved to: {filepath}")




if __name__ == "__main__":
    print("Enhanced utils.py loaded successfully!")
    print("Available functions:")
    print("- CheckpointManager: Advanced checkpoint management")
    print("- compute_comprehensive_metrics: Full metrics calculation")
    print("- print_metrics_summary: Pretty metric printing")
    print("- preprocess_text: Enhanced text preprocessing")
    print("- encode_text: Robust text encoding")