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
all_text = pd.read_csv('/Users/riccimason99/Downloads/Dissertation_2024/all_text_data_frame_clean.csv')


############
########################
# Sentiment Analysis  
########################
############

# Set sentminet analizer 
analyzer = SentimentIntensityAnalyzer()

# get scores for each piece of text and append to list 
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


# save data frame 
#all_text.to_csv('/Users/riccimason99/Downloads/Dissertation_2024/all_text_data_frame_clean.csv', index=False)


# THIS PLOT DOESNT MAKE ANY SENSE, IT IS PLOTTING THE SAME THING TWICE
# Create the plot
jitter = 0.04  # Adjust the jitter amount as needed
all_text['pred_prob_jittered'] = all_text['pred_prob'] + np.random.normal(0, jitter, all_text['pred_prob'].shape)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='neg', y='pred_prob_jittered', data=all_text, label='Actual', color='blue', alpha=0.5)
sns.lineplot(x='neg', y='pred_prob', data=all_text, label='Predicted Probability', color='red')

plt.xlabel('Negative Sentiment Score')
plt.ylabel('Probability of Non-Peaceful Protest')
plt.title('Logistic Regression: Predicted Probability vs. Negative Sentiment Score')
plt.legend()
plt.show()


# There is a statisticallly significant relationship between 
# negative sentiment score and likleyhood of non-peacful in protest.

# Convert to odds ratio
logit = 3.6508
odds_ratio = math.exp(3.6508)  ## Convert to odds ratio 0.3750109700398499
# Get percent probabitliy 
(odds_ratio-1)*100
# So a one unit increase in neg score(neg score is from 0-1) is associate with 
#   an estimated 


all_text.date_of_protest.max()


############
########################
# ML CLASSIFICATION 
########################
############

X = all_text[['compounds', 'neg', 'pos', 'neu']]
y = all_text['binary']
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = .2)

#clf = LogisticRegression()                 #73% Accuracy
#clf = LinearDiscriminantAnalysis()         #78% Accuracy
#clf = KNeighborsClassifier(n_neighbors=3) #84% Accuracy
#clf = SVC(kernel='rbf')                  #73% Accuracy
clf = GaussianNB()                       #86% Accuracy


# Use cross validation to select model and parameters  
kfold = KFold(n_splits = 5, shuffle = True, random_state = 420)
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

