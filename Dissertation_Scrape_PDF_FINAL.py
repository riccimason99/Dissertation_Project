# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import PyPDF2
import os
import pandas as pd



###############################################################################
#####################
###############
###### SCRAPE TEXT FROM PDF
##############
###############################################################################

def get_pdf_files(directory):
    pdf_files = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.pdf', '.PDF')):
            pdf_files.append(filename)
    return pdf_files

# Directory containing PDF files
directory = "/Users/riccimason99/Downloads/Peafcul Articles"

# Get the list of PDF files in the directory
pdf_files = get_pdf_files(directory)

# Print the list of PDF files
print("PDF files in the directory:")
for pdf_file in pdf_files:
    print(pdf_file)


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


# Get the list of PDF files in the directory
pdf_files = [os.path.join(directory, file) for file in get_pdf_files(directory)]

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
df = pd.DataFrame({'File Name': pdf_names, 'Text': pdf_texts})



###############################################################################
#####################
###############
###### Match articles to protest in ACLED
##############
#####################
###############################################################################



# Create New Column for the date of the last article plus 1 day

# initiate empty list
date_plus_1 = []
# loop through file name, which is the date of last article
for i in df["File Name"]:
    date_plus_1.append(i[0:10])
# make the list a column in the data frame
df['date_plus_1'] = date_plus_1

#make new string into date time
df['date_of_protest'] = pd.to_datetime(df['date_plus_1'], format = '%d:%m:%Y')
print(df['date_of_protest'])

# Add one day so the date is the date of the actual protest
df['date_of_protest'] = df['date_of_protest'] + pd.Timedelta(days=1)
print(df['date_of_protest'])
    
#create a date time column for "peace" data frame
peace["date_time"] = pd.to_datetime(peace.event_date, format = '%d %B %Y')


peace_new = df.merge(peace, left_on = 'date_of_protest', right_on = 
                        'date_time', how = 'left')

peace.tags.tail()








    



