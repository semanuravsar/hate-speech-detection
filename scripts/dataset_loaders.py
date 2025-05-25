import torch
from torch.utils.data import Dataset
import pandas as pd

import sys
import os

from scripts.utils import preprocess_text, encode_text

class LatentHatredDataset(Dataset):
    def __init__(self, path, split="train", val_ratio=0.2, random_state=42):
        df = pd.read_csv(path)

        if split == "train" or split == "val":
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(
                df, test_size=val_ratio, random_state=random_state, stratify=df["label_id"]
            )
            df = train_df if split == "train" else val_df
        elif split == "test":
            pass
        else:
            raise ValueError("split must be one of: 'train', 'val', or 'test'")

        self.texts = [preprocess_text(text) for text in df["text"]]
        self.labels = df["label_id"].tolist()

    def __getitem__(self, idx):
        enc = encode_text(self.texts[idx])
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

class StereoSetDataset(Dataset):
    def __init__(self, path, split="train", val_ratio=0.2, random_state=42):
        df = pd.read_csv(path)

        if split == "train" or split == "val":
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(
                df, test_size=val_ratio, random_state=random_state, stratify=df["label"]
            )
            df = train_df if split == "train" else val_df
        elif split == "test":
            pass
        else:
            raise ValueError("split must be one of: 'train', 'val', or 'test'")

        self.texts = [preprocess_text(r['statement']) for _, r in df.iterrows()]
        self.labels = df["label"].tolist()

    def __getitem__(self, idx):
        enc = encode_text(self.texts[idx])
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

class ISarcasmDataset(Dataset):
    def __init__(self, path, split="train", val_ratio=0.2, random_state=42):
        df = pd.read_csv(path)

        if split == "train" or split == "val":
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(
                df, test_size=val_ratio, random_state=random_state, stratify=df["label"]
            )
            df = train_df if split == "train" else val_df
        elif split == "test":
            pass  # assume the file is already the correct split
        else:
            raise ValueError("split must be one of: 'train', 'val', or 'test'")

        self.texts = [preprocess_text(text) for text in df["text"]]
        self.labels = df["label"].astype(int).tolist()

    def __getitem__(self, idx):
        enc = encode_text(self.texts[idx])
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

class ImplicitFineHateDataset(Dataset):
    def __init__(self, path, split="train", val_ratio=0.2, random_state=42):
        df = pd.read_csv(path)

        if split == "train" or split == "val":
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(
                df, test_size=val_ratio, random_state=random_state, stratify=df["label_id"]
            )
            df = train_df if split == "train" else val_df
        elif split == "test":
            pass  # use test split directly
        else:
            raise ValueError("split must be one of: 'train', 'val', or 'test'")

        self.texts = [preprocess_text(text) for text in df["text"]]
        self.labels = df["label_id"].astype(int).tolist()

    def __getitem__(self, idx):
        enc = encode_text(self.texts[idx])
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)
