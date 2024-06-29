#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 22:20:17 2024

@author: riccimason99
"""

import pandas as pd
import numpy as np
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score



# Set sentminet analizer 
analyzer = SentimentIntensityAnalyzer()

# get compund polarity for each piece of text and apply it 
compounds = []
neg = []
pos = []
neu = []

# apply analizer to each piece of text and get compund value
for text in all_text['clean_text']:
    score = analyzer.polarity_scores(text)
    compounds.append(score['compound'])
    neg.append(score['neg'])
    pos.append(score['pos'])
    neu.append(score['neu'])

    
# incude it in data frame 
all_text['neg'] = neg
all_text['compounds'] = compounds
all_text['pos'] = pos
all_text['neu'] = neu



# RUN STATISTICAL ANALYSIS BINARY LOGISTIC REGRESSION 
model = sm.Logit(all_text.binary, all_text.compounds)
results = model.fit()
print(results.summary())

# It looks like there is a statisticallly significant relationship between sentiment score
    # and likleyhood of violence in protest 


# Lets try to use these as feature in a classifier 

X = all_text[['compounds', 'neg', 'pos', 'neu']]
y = all_text['binary']
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = .2)

clf = LogisticRegression()

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))


# Use cross validation 
kfold = KFold(n_splits = 5, shuffle = True, random_state = 420)

scores = cross_val_score(ran_for, X, y, cv = kfold, scoring = 'accuracy')

print(f'Cross-validated Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})')







