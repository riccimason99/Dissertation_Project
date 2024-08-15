#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 22:20:17 2024

@author: riccimason99
"""

import pandas as pd
import numpy as np
import math
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns



# Import data
all_text = pd.read_csv('the data.csv')

############
########################
# Sentiment Analysis  
########################
############

# first untokenize text so it can be used easily for sentiment analysis
all_text['untokenized'] = all_text['tokens'].apply(lambda tokens: ''.join(tokens).replace("', '", ' ').replace("['", '').replace("']", ''))


# Set sentminet analizer 
analyzer = SentimentIntensityAnalyzer()

# get scores for each piece of text and append to list 
compounds = []
neg = []
pos = []
neu = []
# apply analizer to each piece of text and get compund value
for text in all_text['untokenized']:
    score = analyzer.polarity_scores(text)
    compounds.append(score['compound'])
    neg.append(score['neg'])
    pos.append(score['pos'])
    neu.append(score['neu'])



# incude scores in data frame 
all_text['neg'] = neg
all_text['compounds'] = compounds
all_text['pos'] = pos
all_text['neu'] = neu




############
########################
# Logistic Regression Analysis  
########################
############

# Add a constant (intercept term) to the model
all_text['const'] = 1

# Fit the logistic regression model
model = sm.Logit(all_text['binary'], all_text[['const', 'neg']])
results = model.fit()

#print(result.summary())


# New colum w predicted probability 
all_text['pred_prob'] = results.predict(all_text[['const','neg']])

# sort by X variable 
all_text = all_text.sort_values('neg')

# There is a relationship between 
# negative sentiment score and likleyhood of non-peacful in protest.


############
########################
# ML CLASSIFICATION 
########################
############

X = all_text[['compounds', 'neg', 'pos', 'neu']]
y = all_text['binary']
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = .2)

#clf = LogisticRegression()                 
#clf = LinearDiscriminantAnalysis()         
#clf = KNeighborsClassifier(n_neighbors=3) 
#clf = SVC(kernel='rbf')                  
clf = GaussianNB()                       


# Use cross validation to select model and parameters  
kfold = KFold(n_splits = 5, shuffle = True)
scores = cross_val_score(clf, X_train, y_train, cv = kfold, scoring = 'accuracy')
print(f'Cross-validated Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})')

# Fit model to all training data and test on test set
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

###
# VISUALIZE w CONFUSION MATRIX
###
# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
# Show the plot
plt.show()

