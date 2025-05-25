import torch
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F # Import for F.cross_entropy

class MultiTaskBERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased", dropout_rate=0.1): # Changed 'dropout' to 'dropout_rate' for clarity
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Define number of labels for clarity (optional, but good practice)
        self.num_labels_main = 3
        self.num_labels_stereo = 3
        self.num_labels_sarcasm = 2
        self.num_labels_implicit_fine = 7 # Ensure this matches your dataset

        self.classifier_main = nn.Linear(hidden_size, self.num_labels_main)
        self.classifier_stereo = nn.Linear(hidden_size, self.num_labels_stereo)
        self.classifier_sarcasm = nn.Linear(hidden_size, self.num_labels_sarcasm)
        self.classifier_implicit_fine = nn.Linear(hidden_size, self.num_labels_implicit_fine)
        
        self.dropout = nn.Dropout(dropout_rate) # Use dropout_rate

    def forward(self, input_ids, attention_mask, task, labels=None): # MODIFICATION: Added labels=None
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Using sequence_output (last_hidden_state) and taking the [CLS] token's representation
        # is often preferred over pooler_output for classification tasks, but pooler_output can also work.
        # If using CLS token:
        # last_hidden_state = outputs.last_hidden_state
        # sequence_output = self.dropout(last_hidden_state[:, 0, :]) # CLS token
        # Using pooler_output as in your original:
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        logits = None
        if task == "main":
            logits = self.classifier_main(pooled_output)
        elif task == "stereo":
            logits = self.classifier_stereo(pooled_output)
        elif task == "sarcasm":
            logits = self.classifier_sarcasm(pooled_output)
        elif task == "implicit_fine":
            logits = self.classifier_implicit_fine(pooled_output)
        else:
            raise ValueError(f"Unknown task: {task}")

        # MODIFICATION: Calculate loss if labels are provided
        loss = None
        if labels is not None and logits is not None:
            # Determine the number of classes for the current task to ensure correct loss calculation
            if task == "main":
                num_classes_task = self.num_labels_main
            elif task == "stereo":
                num_classes_task = self.num_labels_stereo
            elif task == "sarcasm":
                num_classes_task = self.num_labels_sarcasm
            elif task == "implicit_fine":
                num_classes_task = self.num_labels_implicit_fine
            else: # Should not happen due to earlier check
                num_classes_task = logits.shape[-1]

            loss_fct = nn.CrossEntropyLoss()
            # Ensure logits and labels have compatible shapes.
            # logits: (batch_size, num_classes_task)
            # labels: (batch_size)
            loss = loss_fct(logits.view(-1, num_classes_task), labels.view(-1))
        
        # MODIFICATION: Return structure
        if loss is not None:
            return loss, logits
        else:
            # If no labels provided (e.g., during inference without loss), just return logits
            return logits # For the HPO script's simple evaluate, or if train.py calls without labels