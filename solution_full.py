# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:38:17 2023

@author: vandi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import RobertaTokenizer,RobertaForSequenceClassification
import torch
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to calculate TFIDF Cosine Similarity
def calculate_tfidf_cosine_similarity(target_abstract, prior_art_list):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([target_abstract] + prior_art_list)
    if len(prior_art_list)==0:
        return 0
    else:
        cosine_similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:])
        return np.mean(cosine_similarities.flatten().tolist())

# Function to calculate Jaccard Similarity
def calculate_jaccard_similarity(target_abstract, prior_art_list):
    target_tokens = set(target_abstract.split())
    if len(prior_art_list)==0:
        return 0
    else:
        jaccard_similarities = [len(target_tokens.intersection(set(art.split()))) / len(target_tokens.union(set(art.split()))) for art in prior_art_list]
        return np.mean(jaccard_similarities)
    
# Function to initialise the model for embeddings
def init_dev(model_type):
    if model_type=='roberta':
        # RoBERTa
        legal_roberta_model_name = "roberta-base"
        tokenizer = RobertaTokenizer.from_pretrained(legal_roberta_model_name)
        model = RobertaForSequenceClassification.from_pretrained(legal_roberta_model_name, num_labels=2).to(device)
        
    elif model_type=='legal':
        # Legal-Bert
        legal_bert_model_name = "nlpaueb/legal-bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(legal_bert_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(legal_bert_model_name).to(device)
    elif model_type=='baseline':
        # BERT
        legal_roberta_model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(legal_roberta_model_name)
        model = BertModel.from_pretrained(legal_roberta_model_name).to(device)
    return(tokenizer,model)


# Function to get BERT embeddings for abstract
def get_bert_embeddings(abstract):
    inputs = tokenizer(abstract, return_tensors="pt", max_length=512, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # different outputs for legal-bert
    if model_type=='legal' or model_type=='roberta':
        embeddings = torch.mean(outputs.logits).squeeze().cpu().numpy()
    else:
        embeddings = torch.mean(outputs.last_hidden_state).squeeze().cpu().numpy()
    return embeddings


def removal_spl_stopwords(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Perform stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    # Join tokens back into a string
    processed_text = ' '.join(tokens)
    
    return processed_text

def preprocess_features(df):
    # Sort DataFrame by filing date
    df = df.sort_values(by='filing_date')
    
    # preprocess claims and abstracts
    df['abstract'] = df['abstract'].apply(removal_spl_stopwords)
    df['claims'] = df['claims'].apply(removal_spl_stopwords)
    # Extract abstracts and labels
    abstracts = df['abstract'].tolist()
    labels = df['decision'].tolist()
    claims = df['claims'].tolist()
    
    #Create TFIDF Cosine Similarity and Jaccard Similarity features
    tfidf_cosine_similarities = []
    jaccard_similarities = []
    print("calculating cosine and jaccard similarities\n")
    for i in tqdm(range(0,len(df))):
        prior_art_list = abstracts[:i]
        tfidf_cosine_sim = calculate_tfidf_cosine_similarity(abstracts[i], prior_art_list)
        jaccard_sim = calculate_jaccard_similarity(abstracts[i], prior_art_list)

        tfidf_cosine_similarities.append(tfidf_cosine_sim)
        jaccard_similarities.append(jaccard_sim)

    # Get BERT embeddings for abstracts
    print("getting abstract embeddings\n")
    abstract_embeddings = np.array([get_bert_embeddings(abstract) for abstract in tqdm(abstracts)])
    print("\ngetting claims embeddings\n")
    claims_embeddings = np.array([get_bert_embeddings(claim) for claim in tqdm(claims)])
    # Combine features
    features = np.column_stack((abstract_embeddings, claims_embeddings, tfidf_cosine_similarities, jaccard_similarities))

    return features, labels


def train_model(train_features, train_labels, test_features, test_labels):

    # Split data into training and testing sets
    # Note: If you already have separate training and testing datasets, you can skip this part

    # Train XGBoost model
    model = XGBClassifier(random_state=42)
    model.fit(train_features, train_labels)
    
    # save the model
    model_filename = 'xgboost_model_roberta.model'
    model.save_model(model_filename)
    # Evaluate the model
    accuracy = model.score(test_features, test_labels)
    print(f"Model Accuracy: {accuracy}")
    

model_type ='roberta'
tokenizer,model = init_dev(model_type)

# getting train features and concatenation
train_set = pd.read_csv('train_sample.csv')

# filtering out pending applications
train_set = train_set[train_set["decision"]!=2]

# Pre-processing the features
train_features, train_labels = preprocess_features(train_set)
column_name = ["abstract_embedding","claims_embedding","tfidf_cos","jaccard"]
df_train = pd.DataFrame(train_features, columns=column_name)
df_train.columns = ['tfidf_cos', 'jaccard','abstract_embedding','claims_embedding']
df_train.to_csv('/content/drive/My Drive/train_features.csv',index=False)
df_train_labels = pd.DataFrame(train_labels, columns=["decision"])
df_train_labels.to_csv("train_labels.csv", index=False)

# getting test features
test_set = pd.read_csv('/content/drive/My Drive/test_sample.csv')[:10]

# filtering out pending applications
test_set = test_set[test_set["decision"]!=2]

# Pre-processing the features
test_features, test_labels = preprocess_features(test_set)
column_name_test = ["abstract_embedding","claims_embedding","tfidf_cos","jaccard"]
df_test = pd.DataFrame(test_features, columns=column_name_test)
df_test_labels = pd.DataFrame(test_labels, columns=["decision"])
df_test.to_csv("test_features.csv", index=False)
df_test_labels.to_csv("test_labels.csv", index=False)

# training the model
train_features = df_train[["abstract_embedding","claims_embedding","tfidf_cos","jaccard"]]
test_features = df_test[["abstract_embedding","claims_embedding","tfidf_cos","jaccard"]]
train_labels = df_train_labels["decision"]
test_labels = df_test_labels["decision"]
train_model(train_features, train_labels, test_features, test_labels)
