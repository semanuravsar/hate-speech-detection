import torch
import torch.nn as nn
from transformers import BertModel

class SingleTaskBERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased", dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Linear(hidden_size, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)

        
        return self.classifier(pooled)