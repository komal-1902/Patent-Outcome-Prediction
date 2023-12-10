# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 00:16:05 2023

@author: vandi
"""

import pandas as pd
from pprint import pprint
# Datasets load_dataset function
#from datasets import load_dataset
# Transformers Autokenizer
from transformers import AutoTokenizer
import torch
# Standard PyTorch DataLoader
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams
import numpy as np
from transformers import BertModel,BertTokenizer
import xgboost as xgb
from sklearn.metrics import accuracy_score

# change file name here
df_test = pd.read_csv('/content/drive/My Drive/nlp_test.csv')
df_train = pd.read_csv('/content/drive/My Drive/nlp_train.csv')

#df_test_toy=df_test.iloc[:5]
df_test['filing_date'] = pd.to_datetime(df_test['filing_date'], format='%Y%m%d')
df_train['filing_date'] = pd.to_datetime(df_test['filing_date'], format='%Y%m%d')

def prior_art(row,df):
    file_dt=row
    prior=df[df['filing_date']<file_dt]['abstract'].to_list()
    return prior

def extract_bert_embeddings(text,tokenizer,model):
    # Load pre-trained BERT model and tokenizer
    

    # Tokenize the text and obtain token embeddings
    input_ids = tokenizer(text, return_tensors='pt')['input_ids']
    with torch.no_grad():
        outputs = model(input_ids.cuda())
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Using mean pooling for sentence embeddings

    return embeddings.cpu().numpy()

def prior_embeddings(row,tokenizer,model):
    return [extract_bert_embeddings(abstract,tokenizer,model) for abstract in row]

def cosine_sim(row):
    #print(row)
    return [cosine_similarity(row['embedding_abs'], emb)[0][0] for emb in row["embedding_abs_prior"]]

def avg_dist(row):
    return [sum(abs(row['embedding_abs'] - emb)) / len(emb) for emb in row["embedding_abs_prior"]]

tfidf_vectorizer = TfidfVectorizer()
def tfidf_cos(row):
  if row["abs_prior"]:
    corpus_tfidf = tfidf_vectorizer.fit_transform(row["abs_prior"])
    current_patent_tfidf = tfidf_vectorizer.transform([row["abstract"]])

    return [cosine_similarity(current_patent_tfidf, tfidf)[0][0] for tfidf in corpus_tfidf]
  else:
    return []

def jaccard(row):
    ngram_range = (1, 2)  # Adjust the range based on your preference
    current_patent_ngrams = set(ngrams(row["abstract"].split(), n=ngram_range[0]))
    ngram_jacc = [len(set(ngrams(abstract.split(), n=ngram_range[0]))) for abstract in row["abs_prior"]]
    ngram_jacc=np.array(ngram_jacc)
    ngram_jacc=ngram_jacc / len(current_patent_ngrams)
    return ngram_jacc

def list_each_row(lst):
    return lst

def init_dev():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)
    return(tokenizer,model)

# feature engineering
tokenizer,model=init_dev()
df_train['abs_prior']=df_train['filing_date'].apply(lambda row: prior_art(row,df_train))
df_train['embedding_abs_prior']=df_train['abs_prior'].apply(lambda row: prior_embeddings(row,tokenizer,model))
df_train['embedding_abs']=df_train['abstract'].apply(lambda row: extract_bert_embeddings(row,tokenizer,model))
df_train["feature_cos"]=df_train.apply(lambda row: cosine_sim(row),axis=1)
df_train["feature_dist"]=df_train.apply(lambda row: avg_dist(row),axis=1)
df_train["feature_tfidf_cos"]=df_train.apply(lambda row: tfidf_cos(row),axis=1)
df_train["feature_ngram_jacc"]=df_train.apply(lambda row: jaccard(row),axis=1)

corpus_prior_train=df_train["abstract"].to_list()
corpus_prior_emb=prior_embeddings(corpus_prior_train,tokenizer,model)

df_test['abs_prior']=df_test.apply(lambda row: list_each_row(corpus_prior_train),axis=1)
df_test['embedding_abs_prior']=df_test.apply(lambda row: list_each_row(corpus_prior_emb),axis=1)
df_test['embedding_abs']=df_test['abstract'].apply(lambda row: extract_bert_embeddings(row,tokenizer,model))
df_test["feature_cos"]=df_test.apply(lambda row: cosine_sim(row),axis=1)
df_test["feature_dist"]=df_test.apply(lambda row: avg_dist(row),axis=1)
df_test["feature_tfidf_cos"]=df_test.apply(lambda row: tfidf_cos(row),axis=1)
df_test["feature_ngram_jacc"]=df_test.apply(lambda row: jaccard(row),axis=1)

# calculate mean for all features as xgboost doesn't take list of numbers as input
df_train["mean_feature_cos"]=df_train["feature_cos"].apply(lambda x: np.mean(x))
df_train["mean_feature_dist"]=df_train["feature_dist"].apply(lambda x: np.mean(x))
df_train["mean_feature_tfidf_cos"]=df_train["feature_tfidf_cos"].apply(lambda x: np.mean(x))
df_train["mean_feature_ngram_jacc"]=df_train["feature_ngram_jacc"].apply(lambda x: np.mean(x))


df_test["mean_feature_cos"]=df_test["feature_cos"].apply(lambda x: np.mean(x))
df_test["mean_feature_dist"]=df_test["feature_dist"].apply(lambda x: np.mean(x))
df_test["mean_feature_tfidf_cos"]=df_test["feature_tfidf_cos"].apply(lambda x: np.mean(x))
df_test["mean_feature_ngram_jacc"]=df_test["feature_ngram_jacc"].apply(lambda x: np.mean(x))

# classification
X_train=df_train[["mean_feature_cos","mean_feature_dist","mean_feature_tfidf_cos","mean_feature_ngram_jacc"]]
y_train=df_train["decision"]

X_test=df_test[["mean_feature_cos","mean_feature_dist","mean_feature_tfidf_cos","mean_feature_ngram_jacc"]]
y_test=df_test["decision"]

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'num_class':6,
    'eta': 0.1,
    'max_depth': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}

# Train the XGBoost model
num_rounds = 100
model = xgb.train(params, dtrain, num_rounds)

# Make predictions on the test set
y_pred_proba = model.predict(dtest)
y_pred = y_pred_proba.argmax(axis=1)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy:.2f}")


