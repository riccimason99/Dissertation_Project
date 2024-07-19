#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 22:17:49 2024

@author: riccimason99
"""

###############################################################################
#####################
###############
###### Match articles to protest in ACLED and Merge Data
##############
#####################
###############################################################################

# MERGE VIOLENT PROTESTS W TEXT

# Merge the ACLED data with the scraped text based on dates, if there are no protest 
# for a the date of a given group of text it will appear NA
merged_violent = vil_text.merge(violent_essential, left_on = 'date_of_protest', right_on = 
                        'date_time', how = 'left')
# remove un necessary columns
merged_violent = merged_violent.drop(columns= ['File Name', "date_plus_1", "date_time"])


# MERGE PEACFUL PROTESTS W TEXT