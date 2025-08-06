import re
from transformers import BertTokenizer
import torch
import os

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

import re
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "<url>", text)
    text = re.sub(r"@\w+", "<user>", text)
    text = re.sub(r"#\w+", "<hashtag>", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII
    text = re.sub(r"\s+", " ", text).strip()
    return text

def encode_text(text, max_len=128):
    return tokenizer(text, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")


def save_checkpoint(model, optimizer, epoch, path="checkpoint.pt"):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }, path)

def load_checkpoint(model, optimizer=None, path="checkpoint.pt"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0)