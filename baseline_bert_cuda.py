# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 16:23:26 2023

@author: vandi
"""

import time
from pprint import pprint
# Datasets load_dataset function
from datasets import load_dataset
# Transformers Autokenizer
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
#tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
# Standard PyTorch DataLoader
from torch.utils.data import DataLoader
import pandas as pd
from transformers import BertTokenizer,  BertForSequenceClassification
import torch
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

#dataset_dict = load_dataset('HUPD/hupd',name='sample',data_files="https://huggingface.co/datasets/HUPD/hupd/blob/main/hupd_metadata_2022-02-22.feather", icpr_label=None,force_extract=True,train_filing_start_date='2016-01-01',train_filing_end_date='2016-09-30',val_filing_start_date='2016-10-01',val_filing_end_date='2016-12-31')

start_time = time.time()

print('Loading is done!')

# Label-to-index mapping for the decision status field
#decision_to_str = {'REJECTED': 0, 'ACCEPTED': 1, 'PENDING': 2, 'CONT-REJECTED': 3, 'CONT-ACCEPTED': 4, 'CONT-PENDING': 5}

# Helper function
def map_decision_to_string(example):
    return {'decision': decision_to_str[example['decision']]}

#train_set = dataset_dict['train'].map(map_decision_to_string)
#test_set=dataset_dict['validation'].map(map_decision_to_string)

# CSV:
#pd.DataFrame(train_set).to_csv('data/train.csv', index=False)
#pd.DataFrame(test_set).to_csv('data/test.csv', index=False)
#print("Saved CSVs")

train_set = pd.read_csv("data/train_sample.csv")
test_set = pd.read_csv("data/test_sample.csv")

print("Read Training Data")
print("Start Date:" + str(train_set['filing_date'].min()))
print("End Date:" + str(train_set['filing_date'].max()))
print("Num Records: " + str(train_set.shape[0]))
print("\nRead Test Data")
print("Start Date:" + str(test_set['filing_date'].min()))
print("End Date:" + str(test_set['filing_date'].max()))
print("Num Records: " + str(test_set.shape[0]))

abstracts = train_set['abstract'].to_list()
labels = train_set['decision'].to_list()

abstracts_test = test_set['abstract'].to_list()
labels_test = test_set['decision'].to_list()

#model_name = 'bert-base-uncased'
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#model_name = 'nlpaueb/legal-bert-base-uncased'
#token = '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'
#tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=token)

model_name = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(model_name)

# Tokenize and preprocess text
tokenized_inputs = tokenizer(abstracts, padding=True, truncation=True, return_tensors='pt')
tokenized_test_inputs=tokenizer(abstracts_test, padding=True, truncation=True, return_tensors='pt')

# Extract input_ids, attention_mask, and labels
input_ids = tokenized_inputs['input_ids']
attention_mask = tokenized_inputs['attention_mask']

test_input_ids = tokenized_test_inputs['input_ids']
test_attention_mask = tokenized_test_inputs['attention_mask']

# Convert labels to PyTorch tensor
labels = torch.tensor(labels)
labels_test = torch.tensor(labels_test)

dataset = TensorDataset(input_ids, attention_mask, labels)

# Split the dataset
#train_texts, val_texts, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2, random_state=42)
train_texts = input_ids
val_texts = test_input_ids

train_labels = labels
val_labels = labels_test

# Load pre-trained BERT model
# Normal: bert-base-uncased
# LegalBERT: legalbert-base-uncased
#model = BertForSequenceClassification.from_pretrained(model_name, num_labels=6)
#model = AutoModelForSequenceClassification.from_pretrained(model_name, use_auth_token=token, num_labels=6)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=6)
if torch.cuda.is_available():
    print("Switching to Cuda....")
    model = model.cuda()
    train_texts = train_texts.cuda()
    val_texts = val_texts.cuda()
    train_labels = train_labels.cuda()
    val_labels = val_labels.cuda()

# Create TensorDatasets for train and validation sets
train_dataset = TensorDataset(train_texts, train_labels)
val_dataset = TensorDataset(val_texts, val_labels)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 1
batch_size = 8
accumulation_size = 2

for epoch in range(num_epochs):
    model.train()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    batch_counter = 0

    print("Training")
    print(len(train_dataloader))
    for i, batch in enumerate(train_dataloader):
        print(batch_counter)
        batch_counter += 1
        #optimizer.zero_grad()
        outputs = model(input_ids=batch[0], labels=batch[1])
        loss = outputs.loss
        loss.backward()
        #optimizer.step()

        if (i + 1) % accumulation_size == 0:
	        optimizer.step()
	        optimizer.zero_grad()
            
    # Evaluation
    model.eval()
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    print("Evaluation")
    print(len(val_dataloader))
    batch_counter = 0
    
    with torch.no_grad():
        correct = 0
        total = 0                                                                        
        for batch in val_dataloader:
            print(batch_counter)
            batch_counter += 1
            outputs = model(input_ids=batch[0])
            predictions = torch.argmax(outputs.logits, dim=1)
            total += batch[1].size(0)
            correct += (predictions == batch[1]).sum().item()
        
    accuracy = correct / total
    print(f"Accuracy on the test set: {accuracy}")
    with open("output.txt", "w") as file:
        print(accuracy, file=file)

end_time = time.time()
print("Time Taken: " + str((end_time - start_time)/60) + " minutes")
