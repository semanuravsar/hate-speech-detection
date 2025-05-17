import torch
import torch.nn as nn
from transformers import BertModel

class MultiTaskBERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.classifier_main = nn.Linear(hidden_size, 3)    # Hate speech
        self.classifier_stereo = nn.Linear(hidden_size, 3)  # Stereotypical bias

    def forward(self, input_ids, attention_mask, task):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output

        if task == "main":
            return self.classifier_main(pooled)
        elif task == "stereo":
            return self.classifier_stereo(pooled)
        else:
            raise ValueError(f"Unknown task: {task}")
