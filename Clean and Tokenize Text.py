#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 04:18:31 2024

@author: riccimason99
"""
import pandas as pd
import sys
from bs4 import BeautifulSoup
import re
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
#pip install datefinder
import datefinder
import re
from nltk.corpus import stopwords
from collections import Counter
import string


# Load Data
all_text = pd.read_csv('[...]all_text_data_frame.csv')


###############################################################
#####################
##   CLEAN AND TOKENIZE TEXT
#####################
###############################################################



# Create a function
# Function to remove dates and other unwanted patterns using Regex and Datefinder
def remove_dates_and_unwanted_patterns(text):
    # Use datefinder to find and replace dates
    matches = list(datefinder.find_dates(text))
    for match in matches:
        text = text.replace(str(match), '')
    
    # Define regex patterns for additional date formats and unwanted text
    unwanted_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Formats like 12/31/2020, 31-12-2020
        r'\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4}\b',  # Formats like 31 Dec 2020, 12 March 2024
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},\s\d{4}\b',  # Formats like Dec 31, 2020
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # Formats like 2020/12/31, 2020-12-31
        r'\b\d{4}\b',  # Standalone years like 2023
        r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',  # Days of the week
        r'\b\d{1,2}\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\b',  # Formats like 21 October 2023
        r'\bByline:\s.*',  # Byline
        r'\bLength:\s+\w+\b',  # Length
        r'\bSection:\s+\w+\b',  # Section
        r'\bCopyright.*',  # Copyright information
        r'\bGuest\sEssay\b',  # Guest Essay
        r'\bThe\sNew\sYork\sTimes\s-\sInternational\sEdition\b',  # Newspaper name
        r'\bAll\sRights\sReserved\b',  # All Rights Reserved
    ]

    for pattern in unwanted_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)

    # Remove extra spaces and empty lines
    text = re.sub('\s+', ' ', text).strip()

    return text


# Apply the function to the DataFrame
all_text['clean_text'] = all_text['Text'].apply(remove_dates_and_unwanted_patterns)



# Another function to clean more throughly 
def clean_Text(text):
    # Removes any HTML
    text = BeautifulSoup(text, 'lxml').text
    # Remove URL
    text = re.sub(r'http\S+', '', text)
    # Lower case
    text = text.lower()
    # Remove double spaces
    text = re.sub(r' +', ' ', text)
    
    # Remove unwanted patterns
    unwanted_patterns = [
        r'copyright.*?\n',  # Matches lines starting with 'copyright' followed by any characters until a newline
        r'length.*?\n',     # Matches lines starting with 'length' followed by any characters until a newline
        r'dateline.*?\n',   # Matches lines starting with 'dateline' ect...
        r'byline.*?\n',     # ect...
        r'photos.*?\n',     # ect...
        r'load-date.*?\n',  # ect...
        r'end of document', # Matches 'end of document'
        r'copyright.*?$',   # Matches 'copyright' followed by any characters until the end of the string
        r'length.*?$',      # Matches 'length' followed by any characters until the end of the string
        r'dateline.*?$',    # ect...
        r'byline.*?$',      # ect...
        r'photos.*?$',      # ect...
        r'load-date.*?$',   # ect...
    ]
    
    for pattern in unwanted_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove extra spaces and leading/trailing spaces
    text = re.sub(' +', ' ', text).strip()
    
    return text

# Apply Function
all_text['clean_text'] = all_text['Text'].apply(clean_Text)




# Another Function to remove dates using Regex and Datefinder even better
def remove_dates(text):
    # Use datefinder to find and replace dates
    matches = list(datefinder.find_dates(text))
    for match in matches:
        text = text.replace(str(match), '')
    
    # Define regex patterns for additional date formats
    date_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Formats like 12/31/2020, 31-12-2020
        r'\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4}\b',  # Formats like 31 Dec 2020, 12 March 2024
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{1,2},\s\d{4}\b',  # Formats like Dec 31, 2020
        r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',  # Formats like 2020/12/31, 2020-12-31
        r'\b\d{4}\b',  # Standalone years like 2023
        r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',  # Days of the week
        r'\b\d{1,2}\s(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\b',  # Formats like 21 October 2023
    ]

    for pattern in date_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Remove extra spaces
    text = re.sub(' +', ' ', text).strip()

    return text


# Apply the function to the DataFrame
all_text['clean_text'] = all_text['Text'].apply(remove_dates)


#LETS CHECK IT OUT... looks clean :)
print(all_text.clean_text[15])


# lowercase all text
all_text.clean_text = all_text.clean_text.apply(str.lower)


# TOKENIZE TEXT
#create list of stop words 
nltk.download('punkt')
nltk.download('stopwords')
stopwords_list = set(stopwords.words('english'))



# more symbols to delte 
delte =["'s", "''", '--', '“', '”', 'said', "n't", '``','//']


# Create Function to Tokenize Text IGNORING stop words
def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if word in stopwords_list or word in string.punctuation or word in delte: # ignores any stop wordss
                continue
            if len(word) < 2:
                continue
            else: 
                tokens.append(word)          
    return(tokens)

# Apply Function
all_text["tokens"] = all_text.clean_text.apply(tokenize_text)


# Drop useless columns and text that can not be made available for copyright issues
# while leaving tokenized text 
all_text = all_text.drop(['File Name', 'Text', 'clean_text'], axis =1)


# Save data frame 
all_text.to_csv('...',index = False)

