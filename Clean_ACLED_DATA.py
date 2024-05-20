#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 19:57:39 2024

@author: riccimason99
"""

### ACLED KEY: OGz-I3Os*rU5hFozWRng

import pandas as pd
import numpy as np
import re

#import word2number as num
#rom word2number import w2n


# Load and look at data
violent_2017 = pd.read_csv("/Users/riccimason99/Downloads/Dissertation/US Violent Protests.csv")
violent_2017.shape

# Remove smaller values with the following code
# Enhanced function to extract the first number, considering ranges and textual descriptions
def extract_number(text):
    # Check for and handle textual numbers
    if 'a few hundred' in text:
        return 300  # Approximate "a few hundred" as 300 for filtering
    if 'dozens' in text:
        return np.nan  # "Dozens" is indeterminate but likely less than 300, so we'll ignore these
    
    # Find all numbers or ranges in the text
    numbers_or_ranges = re.findall(r'\d+-\d+|\d+', text)
    if numbers_or_ranges:
        # Handle the first found number or range
        first_number_or_range = numbers_or_ranges[0]
        if '-' in first_number_or_range:
            # If it's a range, take the first part
            return int(first_number_or_range.split('-')[0])
        else:
            # Otherwise, just return the number
            return int(first_number_or_range)
    return np.nan  # Return NaN if no numbers found]]]]]]]]]]]]]]]

# Apply the function to the 'tags' column
violent_2017['first_number'] = violent_2017['tags'].apply(extract_number)
# Filter rows: keep rows where first_number is NaN or first_number >= 200
filtered_violent_2017 = violent_2017[(violent_2017['first_number'] >= 200) | (violent_2017['first_number'].isna())]
print(filtered_violent_2017['tags'].unique())
filtered_violent_2017.shape


#### Remove small values manually 
# Example list of values you want to remove from the 'tags' column
values_to_remove = [
    'crowd size=dozens',
    'crowd size=no report',
    'crowd size=about a dozen',
    'crowd size=small',
    'crowd size=no report; local administrators',
    'crowd size=around a dozen',
    'counter-demonstration; crowd size=no report',
    'counter-demonstration; crowd size=dozens',
    'crowd size=four',
    'crowd size=more than hundred',
    'crowd size=eighteen',
    'crowd size=five',
    'crowd size=two',
    'crowd size=more than four',
    'crowd size=one',
    'crowd size=over a dozen',
    'counter-demonstration; crowd size=more than several',
    'counter-demonstration; crowd size=several',
    'crowd size=seven',
    'crowd size=a large group',
    'crowd size=a dozen',
    'crowd size=three',
    'crowd size=dozens to hundreds',
    'crowd size=thirty',
    'crowd size=four',
    'crowd size=more than hundred',
    'crowd size=eighteen',
    'crowd size=five',
    'crowd size=two',
    'crowd size=more than four',
    'crowd size=one',
    'crowd size=over a dozen',
    'counter-demonstration; crowd size=more than several',
    'counter-demonstration; crowd size=several',
    'crowd size=seven',
    'crowd size=a large group',
    'crowd size=a large group; suggested agents provocateurs',
    'crowd size=some',
    'crowd size=several groups; suggested agents provocateurs',
    'crowd size=several', 'crowd size=at least two',
    'armed; crowd size=no report',
    'armed; counter-demonstration; crowd size=no report',
    'crowd size=six',
    'car ramming; crowd size=no report',
    'armed; counter-demonstration; crowd size=more than three',
    'crowd size=a few',
    'crowd size=a few dozen',
    'armed; counter-demonstration; crowd size=dozens',
    'armed; armed presence; counter-demonstration; crowd size=several dozen',
    'crowd size=dozens to about 150',
    'crowd size=large group',
    'crowd size=about three dozen',
    'counter-demonstration; crowd size=several dozen',
    'crowd size=at least five',
    'counter-demonstration; crowd size=no report; stop the steal',
    'crowd size=at least three; women targeted: girls',
    'crowd size=more than a dozen',
    'crowd size=no report; women targeted: girls',
    'crowd size=eight',
    'crowd size=no report; statue',  'crowd size=over two dozen',
     'crowd size=a crowd',
     'crowd size=a group',
     'counter-demonstration; crowd size=a large group',
     'counter-demonstration; crowd size=more than a few dozen',
     'crowd size=no report; suggested agents provocateurs',
     'armed; counter-demonstration; crowd size=several dozen',
     'crowd size=a small group; statue',
     'armed; armed presence; crowd size=no report',
     'crowd size=a large group; detentions',
     'armed; counter-demonstration; crowd size=more than a few dozen',
     'crowd size=no size; statue',
     'armed; crowd size=dozens; local administrators',
     'crowd size=a crowd; statue',
     'crowd size=at least dozens; statue',
     'counter-demonstration; crowd size=a crowd',
     'crowd size=no report', 'crowd size=dozens', 'crowd size=large' 'crowd size=around 20']

# Remove rows we dont want
filtered_violent_2017 = filtered_violent_2017[~filtered_violent_2017['tags'].isin(values_to_remove)]
print(filtered_violent_2017.tags.unique())
filtered_violent_2017.shape

# Drop some more observations
words_to_drop = ['no report', 'couple', 'a hundred', 'dozen', 'dozens', 'large', 'dozens', 'scores']
# Use DataFrame.apply() with a lambda function to check if any of the words_to_drop are in the 'tags' column
mask = filtered_violent_2017['tags'].apply(lambda x: any(word in x for word in words_to_drop))
# Invert the mask to keep rows that do not contain any of the words_to_drop
violent_protests = filtered_violent_2017[~mask]
violent_protests.shape    


# Drop columns that I do not care about
violent_protests.columns
violent_protests = violent_protests.drop(['time_precision', 
        'sub_event_type', 'first_number', 'event_type', 'actor1',  'inter1', 
        'actor2', 'timestamp', 'region', 'geo_precision',  'source_scale', 'iso', 
        'inter2', 'interaction', 'event_id_cnty', 'civilian_targeting',
        'admin3', 'country', 'disorder_type'], axis  = 1)

violent_protests.reset_index(drop = True, inplace = True)
violent_protests.shape
violent_protests.tags.unique()


## Lets see what these observations are all about
pd.set_option('display.max_colwidth', 1000)
print(violent_protests.notes.tail())
violent_protests.shape


##############################################################################################################
##############################################################################################################
#  NON VIOLENT
##############################################################################################################
##############################################################################################################


data_peace = pd.read_csv("/Users/riccimason99/Downloads/Dissertation/US  Non-Violent Protests .csv")
print(data_peace.tags.head(50))
print(data_peace.tags.unique())
data_peace.shape # 59564 observations


# REMOVE VALUES OF SMALL PROTEST

## This code removes a lot of values which the first number is below 200
def check_number(s):
    # Ensure the input is a string
    if isinstance(s, str):
        # Search for numbers in the string
        numbers = re.findall(r'\d+', s)
        if numbers:
            # Convert the first found number to integer
            num = int(numbers[0])
            # Return False if the number is less than 200 (indicating removal)
            return num >= 200
    # Return True if no number is found or the input is not a string (indicating retention)
    return True

# Apply the function to the 'tags' column and filter the DataFrame
peace_ = data_peace[data_peace['tags'].apply(check_number)]
# Show the filtered DataFrame
peace_.shape #47021 observations # we lost a lot of observatoins

print(peace_.tags.unique())
print(peace_.tags.iloc[200])

# remove values in "values_to_remove"
filtered_peace_ = peace_[~peace_['tags'].isin(values_to_remove)]
filtered_peace_.shape
   
# remove useless columbs 
filtered_peace_ = filtered_peace_.drop(['time_precision', 
        'sub_event_type', 'event_type', 'actor1',  'inter1', 
        'actor2', 'timestamp', 'region', 'geo_precision',  'source_scale', 'iso', 
        'inter2', 'interaction', 'event_id_cnty', 'civilian_targeting',
        'admin3', 'country', 'disorder_type'], axis  = 1)

print(filtered_peace_.tags.unique()[:50])


# Convert 'tags' column to string type and then filter out rows
filtered_peace_['tags'] = filtered_peace_['tags'].astype(str)
filtered_peace_ = filtered_peace_[~filtered_peace_['tags'].str.contains('|'.join(words_to_drop))]
filtered_peace_.shape
# Now filtered_peace_ contains only rows where 'tags' column does not contain any word from words_to_drop

print(filtered_peace_.tags.unique()[:50])
1+1
2+2
print('poop')



