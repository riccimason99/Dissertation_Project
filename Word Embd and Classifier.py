#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 05:20:07 2024

@author: riccimason99
"""

import pandas as pd
import numpy as np

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import nltk
import gensim
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import re
import gensim.models as g


import multiprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import utils
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier  
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from sklearn.metrics import f1_score




############
########################
# PRE-PROCESSING
########################
############

# Import Data
df = pd.read_csv([...]all_text_data_frame_clean.csv')
df = df[['clean_text','binary']]

# Split words by space
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



############
########################
# CREATE WORD EMBEDDINGS USING Doc2Vec
########################
############

# Create tags for documents, (tags will be the binary values 0 or 1 to be used
# as the y for classification models)
train_tagged = df.apply(
    lambda r: TaggedDocument(words=tokenize_text(r['clean_text']), tags=[r.binary]), axis=1)

# Set cores
cores = multiprocessing.cpu_count()

# Create model specifics
model_dbow = Doc2Vec(dm=0, vector_size=300, negative=0, hs=0, min_count=5, sample = 0, workers=cores)
# Build vocabulary 
model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])

# Train the model on training data 
for epoch in range(30):
    model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), total_examples=len(train_tagged.values), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha

# Function creates word embedding which will be used for the classifier 
def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, epochs=20)) for doc in sents])
    return targets, regressors


############
########################
# CLASSIFICATION 
########################
############

# Set the random seed for reproducibility
np.random.seed(24)

# Set classifier
clf = SVC()
# clf = RandomForestClassifier()

# Set input and respnse variables, with "home made" model
y, X = vec_for_learning(model_dbow, train_tagged)


# Use cross validation 
kfold = KFold(n_splits = 3, shuffle = True)

scores = cross_val_score(clf, X, y, cv = kfold, scoring = 'accuracy')

print(f'Cross-validated Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})')

# Validation scores are very low, I will not proceed to do run the model again
# on a test portion of the data 



############
########################
# CREATE WORD EMBEDDINGS USING Word2Vec
########################
############

from gensim.models import KeyedVectors
import os
import gzip


# Path to the downloaded model, you can download it by visiting this site 
# https://github.com/harmanpreet93/load-word2vec-google?tab=readme-ov-file
model_path = '/Users/riccimason99/Downloads/Dissertation_2024/Violent Articles/GoogleNews-vectors-negative300.bin.gz'
# Load the pre-trained model
Word2Vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split (df['tokens'], df['binary'] , test_size=0.2)


# =============================================================================
# # CONVERTING WORDS INTO VECTORS
# =============================================================================

# Convert index_to_key to a set for faster membership checking
words_set = set(Word2Vec_model.index_to_key)

# Function to create vectors 
def sentence_to_vec(sentence, model, words_set):
    return np.array([model[word] for word in sentence if word in words_set])

# Add a progress bar and create vector for X_train
X_train_vect = Parallel(n_jobs=-1)(
    delayed(sentence_to_vec)(sentence, Word2Vec_model, words_set) for sentence in tqdm(X_train, desc="Processing X_train")
)
X_train_vect = np.array([vec for vec in X_train_vect if vec.size > 0])  # Filter out empty vectors


# now do same for X_test
X_test_vect = Parallel(n_jobs=-1)(
    delayed(sentence_to_vec)(sentence, Word2Vec_model, words_set) for sentence in tqdm(X_test, desc="Processing X_test")
)
X_test_vect = np.array([vec for vec in X_test_vect if vec.size > 0])  # Filter out empty vectors



# Compute sentence vectors by averaging the word vectors to make them the same length
X_train_vect_avg = []
for v in X_train_vect:
    if v.size:
        X_train_vect_avg.append(v.mean(axis=0))
    else:
        X_train_vect_avg.append(np.zeros(300, dtype=float))
        
# do the same for x test
X_test_vect_avg = []
for v in X_test_vect:
    if v.size:
        X_test_vect_avg.append(v.mean(axis=0))
    else:
        X_test_vect_avg.append(np.zeros(300, dtype=float))


# Are our sentence vector lengths consistent?
for i, v in enumerate(X_train_vect_avg):
    print(len(X_train.iloc[i]), len(v))


# =============================================================================
# # APPLYING VECTORS TO ML CLASSIFIERS 
# =============================================================================

rf = RandomForestClassifier() #SVC() #
rf_model = rf.fit(X_train_vect_avg, y_train.values.ravel())

scores = cross_val_score(rf_model, X_train_vect_avg, y_train, cv = kfold, scoring = 'f1')
print(f'Cross-validated Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})')





