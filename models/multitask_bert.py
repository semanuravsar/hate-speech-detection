import torch
import torch.nn as nn
from transformers import BertModel

class MultiTaskBERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased", dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.classifier_main = nn.Linear(hidden_size, 3)
        self.classifier_stereo = nn.Linear(hidden_size, 3)
        self.classifier_sarcasm = nn.Linear(hidden_size, 2)
        self.classifier_implicit_fine = nn.Linear(hidden_size, 7)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, task):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)

        if task == "main":
            return self.classifier_main(pooled)
        elif task == "stereo":
            return self.classifier_stereo(pooled)
        elif task == "sarcasm":
            return self.classifier_sarcasm(pooled)
        elif task == "implicit_fine":
            return self.classifier_implicit_fine(pooled)
        else:
            raise ValueError(f"Unknown task: {task}")