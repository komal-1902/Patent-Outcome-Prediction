# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:23:26 2023

@author: vandi
"""

from pprint import pprint
# Datasets load_dataset function
from datasets import load_dataset
# Transformers Autokenizer
from transformers import AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
# Standard PyTorch DataLoader
from torch.utils.data import DataLoader
import pandas as pd
from transformers import BertTokenizer,  BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

dataset_dict = load_dataset('HUPD/hupd',
    name='all',
    data_files="https://huggingface.co/datasets/HUPD/hupd/blob/main/hupd_metadata_2022-02-22.feather", 
    icpr_label=None,
    force_extract=True,
    train_filing_start_date='2010-01-01',
    train_filing_end_date='2016-12-31',
    val_filing_start_date='2017-01-01',
    val_filing_end_date='2018-12-31'
)

print('Loading is done!')

# Label-to-index mapping for the decision status field
decision_to_str = {'REJECTED': 0, 'ACCEPTED': 1, 'PENDING': 2, 'CONT-REJECTED': 3, 'CONT-ACCEPTED': 4, 'CONT-PENDING': 5}

# Helper function
def map_decision_to_string(example):
    return {'decision': decision_to_str[example['decision']]}

train_set = dataset_dict['train'].map(map_decision_to_string)
test_set=dataset_dict['validation'].map(map_decision_to_string)

abstracts = train_set['abstract']
labels = train_set['decision']

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize and preprocess text
tokenized_inputs = tokenizer(abstracts, padding=True, truncation=True, return_tensors='pt')

# Extract input_ids, attention_mask, and labels
input_ids = tokenized_inputs['input_ids']
attention_mask = tokenized_inputs['attention_mask']

# Convert labels to PyTorch tensor
labels = torch.tensor(labels)

dataset = TensorDataset(input_ids, attention_mask, labels)

# Split the dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    input_ids, labels, test_size=0.2, random_state=42
)

# Create TensorDatasets for train and validation sets
train_dataset = TensorDataset(train_texts, train_labels)
val_dataset = TensorDataset(val_texts, val_labels)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 3
batch_size = 16

for epoch in range(num_epochs):
    model.train()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch[0], labels=batch[1])
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

with torch.no_grad():
    correct = 0
    total = 0
    
    for batch in val_dataloader:
        outputs = model(input_ids=batch[0])
        predictions = torch.argmax(outputs.logits, dim=1)
        total += batch[1].size(0)
        correct += (predictions == batch[1]).sum().item()

accuracy = correct / total
print(f"Accuracy on the test set: {accuracy}")