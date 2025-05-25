import torch
from torch.utils.data import Dataset
import pandas as pd
from scripts.utils import preprocess_text, encode_text

class LatentHatredDataset(Dataset):
    def __init__(self, path):
        df = pd.read_csv(path)
#        df = df[df["label_id"] != 2].copy()
#        df["label_id"] = df["label_id"].map({0: 0, 1: 1})
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
    def __init__(self, path):
        df = pd.read_csv(path)
        self.texts = [preprocess_text(r["statement"]) for _, r in df.iterrows()]
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
