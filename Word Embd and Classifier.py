#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 05:20:07 2024

@author: riccimason99
"""
import nltk
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import cross_val_score, KFold
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing


# Import Data
df = pd.read_csv('/Users/riccimason99/Downloads/Dissertation_2024/all_text_data_frame_clean.csv')
df = df[['clean_text','binary']]


df['clean_text'].apply(lambda x: len(x.split(' '))).sum()


# Lowercase and remove symbols/punctuatoin 
from bs4 import BeautifulSoup
def cleanText(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text) 
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    text = text.replace('x', '')
    return text
df['clean_text'] = df['clean_text'].apply(cleanText)

# Split the data into train and test
#train, test = train_test_split(df, test_size=0.2, random_state=42)


# Tokenize and remove stop words, ignore words with one letter 
from nltk.corpus import stopwords
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens


# Set up tokens with a tag, tag will be the binary 0 for peacful 1 for non peacful 
#train_tagged = train.apply(
#    lambda r: TaggedDocument(words=tokenize_text(r['clean_text']), tags=[r.binary]), axis=1)
#test_tagged = test.apply(
#    lambda r: TaggedDocument(words=tokenize_text(r['clean_text']), tags=[r.binary]), axis=1)







train_tagged = df.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['clean_text']), tags=[r.binary]), axis=1)








# Set cores
cores = multiprocessing.cpu_count()

# Create model specifics and build vocabulary 
model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, hs=0, min_count=5, sample = 0, workers=cores)
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

# Train the model on training data 
for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

# Create word embedding which will be used for the classifier 
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, epochs=20)) for doc in sents])
    return targets, regressors

# Set random forest as classifier
ran_for = RandomForestClassifier()

# train random forest classifier
#y_train, X_train = vec_for_learning(model_dbow, train_tagged)
#y_test, X_test = vec_for_learning(model_dbow, test_tagged)

#ran_for.fit(X_train, y_train)
#y_pred = ran_for.predict(X_test)

#print('Testing accuracy %s' % accuracy_score(y_test, y_pred))
#print('Testing F1 score: {}'.format(f1_score(y_test, y_pred, average='weighted')))


y, X = vec_for_learning(model_dbow, train_tagged)


# Use cross validation 
kfold = KFold(n_splits = 5, shuffle = True, random_state = 420)

scores = cross_val_score(ran_for, X, y, cv = kfold, scoring = 'accuracy')

print(f'Cross-validated Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})')






