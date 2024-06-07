# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import PyPDF2
import os
import pandas as pd
from datetime import timedelta


###############################################################################
###############
###### FUNCTIONS TO SCRAPE TEXT FROM PDF
##############
###############################################################################

# Function to scrape names of PDF files from the directory
def get_pdf_files(directory):
    pdf_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.pdf', '.PDF')):
            pdf_files.append(filename)
    return pdf_files

# Print the list of PDF files
#print("PDF files in the directory:")
#for pdf_file in pdf_files:
 #   print(pdf_file)

# Create the function wich scrapes text from PDF 
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text()
            return text
    except Exception as e:
        print(f"Error: {e}")
        return None
    

###############################################################################
###############
###### SCRAPE TEXT FOR PEACFUL PROTESTS
##############
###############################################################################

directory = "/Users/riccimason99/Downloads/Dissertation_2024/Peafcul Articles"


# Use the Previously Created Function to Get File Names

# Get the list of PDF files in the directory
pdf_files = get_pdf_files(directory)
# Print the list of PDF files
#print("PDF files in the directory:")
#for pdf_file in pdf_files:
 #   print(pdf_file)


# Get the list of PDF files for peacful protests
pdf_files = [os.path.join(directory, file) for file in pdf_files]

# Initialize lists to store file names and text
pdf_names = []
pdf_texts = []

# Iterate over each PDF file, extract text, and accumulate it along with the file name
for pdf_file in pdf_files:
    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_file)
    
    # Append the file name to the list
    pdf_names.append(os.path.basename(pdf_file))
    
    # Append the extracted text to the list
    pdf_texts.append(pdf_text)

# Create a DataFrame to store the data
peace_text = pd.DataFrame({'File Name': pdf_names, 'Text': pdf_texts})





###
# Add the date of the protest wich the articles correspond as a column
###
 
# initiate empty list
date_of_protest = []
# loop through file name, which is the date of last article
for i in peace_text["File Name"]:
     date_of_protest.append(i[0:10])
# make the list a column in the data frame
peace_text['date_of_protest'] = date_of_protest


peace_text['date_of_protest'] = pd.to_datetime(peace_text['date_of_protest'], format = '%d:%m:%Y')
#print(vil_text['date_of_protest'])
# Add one day so the date is the date of the actual protest
peace_text['date_of_protest'] = peace_text['date_of_protest'] + pd.Timedelta(days=1)


###############################################################################
###############
###### SCRAPE TEXT FOR NON - PEACFUL PROTESTS
##############
###############################################################################

directory_violent = '/Users/riccimason99/Downloads/Dissertation_2024/Violent Articles'

# Use the Previously Created Function to Get File Names

pdf_violent_files = get_pdf_files(directory_violent)
# Print the list of PDF files
#print("PDF files in the directory:")
#for pdf_file in pdf_violent_files:
 #   print(pdf_file)

# Use the Previously Created Function to Get File Names
pdf_violent_files = [os.path.join(directory_violent, file) for file in pdf_violent_files]

# Initialize lists to store file names and text
pdf_names = []
pdf_texts = []

# Iterate over each PDF file, extract text, and accumulate it along with the file name
for pdf_file in pdf_violent_files:
    # Extract text from the PDF
    pdf_text = extract_text_from_pdf(pdf_file)
    
    # Append the file name to the list
    pdf_names.append(os.path.basename(pdf_file))
    
    # Append the extracted text to the list
    pdf_texts.append(pdf_text)

# Create a DataFrame to store the data
vil_text = pd.DataFrame({'File Name': pdf_names, 'Text': pdf_texts})




#####
# Add the date of the protest wich the articles correspond as a column
#####

# initiate empty list
date_plus_1 = []
# loop through file name, which is the date of last article
for i in vil_text["File Name"]:
    date_plus_1.append(i[0:10])
# make the list a column in the data frame
vil_text['date_plus_1'] = date_plus_1

# Change it to date tiem
vil_text['date_of_protest'] = pd.to_datetime(vil_text['date_plus_1'], format = '%d:%m:%Y')

# Add one day so the date is the date of the actual protest
vil_text['date_of_protest'] = vil_text['date_of_protest'] + pd.Timedelta(days=1)

#drop unneeded column 
vil_text.drop('date_plus_1', axis=1, inplace=True)



###################################
###################################
###############
#   CONCATONATE NON-PEACFUL AND PEACFUL DATA FRAMES 
###############
###################################
###################################

# Create binary tags for peafcul or non peacful 
vil_text['binary'] = 1
peace_text['binary'] = 0

# Concat the data frames 
all_text = pd.concat([peace_text, vil_text])
# Check for duplicate file names
print(all_text[all_text['File Name'].duplicated()])

# Save as CSV
all_text.to_csv('/Users/riccimason99/Downloads/Dissertation_2024/all_text_data_frame.csv',index = False)








###################################
###################################
###############
#   MERGE ACLED DATA WITH ACTUAL PROTEST 
###############
###################################
###################################

# Merge the ACLED data with the scraped text based on dates, if there are no protest 
# for a the date of a given group of text it will appear NA
#merged_peace = df.merge(peace_essential, left_on = 'date_of_protest', right_on = 
#                       'date_time', how = 'left')
# remove un necessary columns
#merger_peace = merged_peace.drop(columns= ['File Name', "date_plus_1", "date_time"])











    



