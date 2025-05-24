import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

import sys
import os

from utils import preprocess_text, encode_text


class LatentHatredDataset(Dataset):
    """
    Enhanced dataset with proper three-way stratified split (70/15/15)
    """
    def __init__(self, path, split="train", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """
        Args:
            path: Path to CSV file
            split: 'train', 'val', 'test', or 'all'
            train_ratio: Proportion for training (default: 0.7)
            val_ratio: Proportion for validation (default: 0.15) 
            test_ratio: Proportion for test (default: 0.15)
            random_state: For reproducibility
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
        
        df = pd.read_csv(path)
        original_size = len(df)
        
        if split == "all":
            # Return full dataset without splitting
            final_df = df
        elif split in ["train", "val", "test"]:
            # Perform stratified three-way split
            from sklearn.model_selection import train_test_split
            
            # First split: separate training from (validation + test)
            train_df, temp_df = train_test_split(
                df, 
                test_size=(val_ratio + test_ratio), 
                random_state=random_state,
                stratify=df["label_id"]
            )
            
            # Second split: separate validation from test
            val_size_in_temp = val_ratio / (val_ratio + test_ratio)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=(1 - val_size_in_temp),
                random_state=random_state,
                stratify=temp_df["label_id"]
            )
            
            # Select appropriate split
            if split == "train":
                final_df = train_df
            elif split == "val":
                final_df = val_df
            else:  # test
                final_df = test_df
        else:
            raise ValueError("split must be one of: 'train', 'val', 'test', 'all'")
        
        # Process data
        self.texts = [preprocess_text(text) for text in final_df["text"]]
        self.labels = final_df["label_id"].tolist()
        
        # Print split information
        self._print_split_info(split, len(self.labels), original_size)
        self._print_class_distribution()
    
    def _print_split_info(self, split, current_size, original_size):
        """Print information about the current split"""
        percentage = (current_size / original_size) * 100
        print(f"📊 {split.capitalize()} set: {current_size:,} samples ({percentage:.1f}% of total)")
    
    def _print_class_distribution(self):
        """Print class distribution for this split"""
        label_counts = pd.Series(self.labels).value_counts().sort_index()
        print("   Class distribution:")
        for label_id, count in label_counts.items():
            percentage = (count / len(self.labels)) * 100
            print(f"     Class {label_id}: {count:,} samples ({percentage:.1f}%)")
    
    def get_class_weights(self):
        """Calculate class weights for handling class imbalance"""
        from sklearn.utils.class_weight import compute_class_weight
        
        unique_labels = np.unique(self.labels)
        class_weights = compute_class_weight(
            'balanced', 
            classes=unique_labels, 
            y=self.labels
        )
        
        return torch.FloatTensor(class_weights)
    
    def __getitem__(self, idx):
        enc = encode_text(self.texts[idx])
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)


def verify_stratification(csv_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """
    Verify that stratification is working correctly across all splits
    """
    print("🔍 Verifying Stratification Across Splits")
    print("=" * 50)
    
    # Load original data
    df = pd.read_csv(csv_path)
    original_dist = df["label_id"].value_counts(normalize=True).sort_index()
    print("Original distribution:")
    for label_id, proportion in original_dist.items():
        print(f"  Class {label_id}: {proportion:.3f}")
    
    # Create all splits
    train_dataset = LatentHatredDataset(csv_path, split="train", 
                                       train_ratio=train_ratio, val_ratio=val_ratio, 
                                       test_ratio=test_ratio, random_state=random_state)
    val_dataset = LatentHatredDataset(csv_path, split="val",
                                     train_ratio=train_ratio, val_ratio=val_ratio, 
                                     test_ratio=test_ratio, random_state=random_state)
    test_dataset = LatentHatredDataset(csv_path, split="test",
                                      train_ratio=train_ratio, val_ratio=val_ratio, 
                                      test_ratio=test_ratio, random_state=random_state)
    
    # Check distributions match
    print(f"\n✅ Stratification verification:")
    splits_data = [
        ("Train", train_dataset.labels),
        ("Val", val_dataset.labels), 
        ("Test", test_dataset.labels)
    ]
    
    for split_name, labels in splits_data:
        split_dist = pd.Series(labels).value_counts(normalize=True).sort_index()
        max_diff = max(abs(split_dist - original_dist))
        print(f"  {split_name}: Max deviation = {max_diff:.4f}")
        if max_diff < 0.05:  # 5% tolerance
            print(f"    ✅ Good stratification")
        else:
            print(f"    ⚠️  Large deviation detected")
    
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Test the dataset splitting
    print("Testing dataset splitting functionality...")
    
    # Example usage
    csv_path = "/home/altemir/hate-speech-detection/datasets/latent_hatred_3class.csv"  # Update this path
    
    try:
        train_ds, val_ds, test_ds = verify_stratification(csv_path)
        print(f"\n📋 Summary:")
        print(f"   Train: {len(train_ds)} samples")
        print(f"   Val: {len(val_ds)} samples") 
        print(f"   Test: {len(test_ds)} samples")
        print(f"   Total: {len(train_ds) + len(val_ds) + len(test_ds)} samples")
        
    except FileNotFoundError:
        print("⚠️  Update the csv_path variable to test the functionality")