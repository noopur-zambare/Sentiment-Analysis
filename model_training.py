#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from custom_tokenizer_function import CustomTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import spacy

nlp = spacy.load('en_core_web_sm')

#Loading the dataset
df = pd.read_csv('xx.tsv',sep='\t') 
dataset = df[['verified_reviews','rating']]
dataset.columns = ['Review', 'Sentiment']

# based on ratings, creating "sentiment" column
def calc_sentiments(labels):
  sentiments = []
  for label in labels:
    if label > 3.0:
      sentiment = 1
    elif label <= 3.0:
      sentiment = 0
    sentiments.append(sentiment)
  return sentiments
  
dataset['Sentiment'] = calc_sentiments(dataset.Sentiment)

# declaring independent and dependent variables
x = dataset['Review']
y = dataset['Sentiment']

custom_tokenizer = CustomTokenizer()

# Vectorization
tfidf = TfidfVectorizer(tokenizer=custom_tokenizer.text_data_cleaning)

# model training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, stratify = dataset.Sentiment, random_state = 0)

model = SGDClassifier(loss="hinge", penalty="l2", max_iter=100)

# vectorization -> classification
pipeline = Pipeline([('tfidf',tfidf), ('clf',model)])
pipeline.fit(x_train, y_train)

import joblib
joblib.dump(pipeline,'model.pkl')

## Model Performance
y_pred = pipeline.predict(x_test)
accuracy=round(accuracy_score(y_test, y_pred)*100,2)
print(accuracy)