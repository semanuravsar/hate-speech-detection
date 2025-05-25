import torch
from torch.utils.data import Dataset
import pandas as pd

import sys
import os

from scripts.utils import preprocess_text, encode_text

class LatentHatredDataset(Dataset):
<<<<<<< Updated upstream
    def __init__(self, path, split="train", val_ratio=0.1, random_state=42):
=======
    def __init__(self, path):
>>>>>>> Stashed changes
        df = pd.read_csv(path)
        
        # Keep only non-hate (0) and implicit hate (1)
        df = df[df["label_id"] != 2].copy()
        df["label_id"] = df["label_id"].map({0: 0, 1: 1})  # Re-map labels to binary

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

class StereoSetDataset(Dataset):
<<<<<<< Updated upstream
    def __init__(self, path, split="train", val_ratio=0.1, random_state=42):
=======
    def __init__(self, path):
>>>>>>> Stashed changes
        df = pd.read_csv(path)

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
<<<<<<< Updated upstream
=======

class ISarcasmDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)

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
    def __init__(self, path):
        df = pd.read_csv(path)

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
>>>>>>> Stashed changes
