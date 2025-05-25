# scripts/dataset_loaders.py

import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split # Import for splitting

# Assuming utils.py is in the same 'scripts' directory or project root is in PYTHONPATH
try:
    from .utils import preprocess_text, encode_text
except ImportError:
    from utils import preprocess_text, encode_text

class BaseTaskDatasetInMemorySplit(Dataset):
    def __init__(self,
                 original_data_path, # Path to the original, unsplit CSV
                 split="train",      # "train", "val", "test", or "all"
                 train_ratio=0.7,    # Proportion for training
                 val_ratio=0.15,     # Proportion for validation
                 # test_ratio is implicitly 1.0 - train_ratio - val_ratio
                 random_state=42,
                 text_col="text",
                 label_col="label_id"):
        """
        Loads the original dataset and performs an in-memory train/val/test split.
        Args:
            original_data_path (str): Path to the original, unsplit CSV file.
            split (str): One of "train", "val", "test", or "all".
            train_ratio (float): Proportion of data for the training set.
            val_ratio (float): Proportion of data for the validation set.
            random_state (int): Seed for reproducible splits.
            text_col (str): Name of the column containing text.
            label_col (str): Name of the column containing labels.
        """
        if not os.path.exists(original_data_path):
            raise FileNotFoundError(f"Original dataset file not found: {original_data_path}")

        df = pd.read_csv(original_data_path)
        original_size = len(df)

        final_df = None
        if split == "all":
            final_df = df
        elif split in ["train", "val", "test"]:
            # Ensure ratios make sense
            if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and (train_ratio + val_ratio) < 1):
                raise ValueError("train_ratio and val_ratio must be between 0 and 1, and their sum less than 1.")

            test_ratio = 1.0 - train_ratio - val_ratio
            if test_ratio <= 0: # Should be caught by sum < 1 but good to be explicit
                 raise ValueError("train_ratio + val_ratio must be less than 1 to leave data for the test set.")


            # First split: separate training from (validation + test)
            stratify_column = df[label_col]
            train_df, temp_df = train_test_split(
                df,
                test_size=(val_ratio + test_ratio), # Size of (val + test)
                random_state=random_state,
                stratify=stratify_column
            )

            # Second split: separate validation from test
            # Calculate val_size relative to the temp_df (which is val + test)
            if (val_ratio + test_ratio) == 0: # Avoid division by zero if test_ratio was huge
                val_size_in_temp = 0
            else:
                val_size_in_temp = val_ratio / (val_ratio + test_ratio)

            if val_size_in_temp == 0 : # No validation set needed or possible
                 val_df = pd.DataFrame(columns=temp_df.columns) # Empty val_df
                 test_df = temp_df
            elif val_size_in_temp == 1: # No test set needed or possible from temp_df
                 val_df = temp_df
                 test_df = pd.DataFrame(columns=temp_df.columns) # Empty test_df
            else:
                stratify_temp_column = temp_df[label_col]
                val_df, test_df = train_test_split(
                    temp_df,
                    test_size=(1.0 - val_size_in_temp), # Size of test relative to temp_df
                    random_state=random_state,
                    stratify=stratify_temp_column
                )

            if split == "train":
                final_df = train_df
            elif split == "val":
                final_df = val_df
            else:  # test
                final_df = test_df
        else:
            raise ValueError("split must be one of: 'train', 'val', 'test', or 'all'")

        if final_df is None or final_df.empty:
            print(f"Warning: The '{split}' split for {original_data_path} resulted in an empty DataFrame. Check ratios and data.")
            self.texts = []
            self.labels = []
        else:
            self.texts = [preprocess_text(str(text)) for text in final_df[text_col]]
            self.labels = final_df[label_col].tolist()

        # Print split information (optional, but good for debugging)
        current_size = len(self.labels)
        percentage = (current_size / original_size) * 100 if original_size > 0 else 0
        print(f"Initialized '{split}' split for {os.path.basename(original_data_path)}: {current_size} samples ({percentage:.1f}% of original).")
        if current_size > 0:
            self._print_class_distribution()


    def _print_class_distribution(self):
        """Print class distribution for this split"""
        if not self.labels:
            print("   Class distribution: No labels to display (empty split).")
            return
        label_counts = pd.Series(self.labels).value_counts().sort_index()
        print("   Class distribution:")
        for label_id, count in label_counts.items():
            percentage = (count / len(self.labels)) * 100
            print(f"     Class {label_id}: {count:,} samples ({percentage:.1f}%)")


    def __getitem__(self, idx):
        enc = encode_text(self.texts[idx])
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)

# --- Task-Specific Dataset Classes ---
# Define standard split ratios for consistency if not overridden
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
# Test ratio is 1.0 - DEFAULT_TRAIN_RATIO - DEFAULT_VAL_RATIO

class LatentHatredDataset(BaseTaskDatasetInMemorySplit):
    def __init__(self, original_data_path, split="train", random_state=42):
        super().__init__(original_data_path, split,
                         train_ratio=DEFAULT_TRAIN_RATIO, val_ratio=DEFAULT_VAL_RATIO,
                         random_state=random_state,
                         text_col="text", label_col="label_id")

class StereoSetDataset(BaseTaskDatasetInMemorySplit):
    def __init__(self, original_data_path, split="train", random_state=42):
        super().__init__(original_data_path, split,
                         train_ratio=DEFAULT_TRAIN_RATIO, val_ratio=DEFAULT_VAL_RATIO,
                         random_state=random_state,
                         text_col="statement", label_col="label")

class ISarcasmDataset(BaseTaskDatasetInMemorySplit):
    def __init__(self, original_data_path, split="train", random_state=42):
        super().__init__(original_data_path, split,
                         train_ratio=DEFAULT_TRAIN_RATIO, val_ratio=DEFAULT_VAL_RATIO,
                         random_state=random_state,
                         text_col="text", label_col="label")

class ImplicitFineHateDataset(BaseTaskDatasetInMemorySplit):
    def __init__(self, original_data_path, split="train", random_state=42):
        super().__init__(original_data_path, split,
                         train_ratio=DEFAULT_TRAIN_RATIO, val_ratio=DEFAULT_VAL_RATIO,
                         random_state=random_state,
                         text_col="text", label_col="label_id")