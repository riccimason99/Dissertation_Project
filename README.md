
# Predicting Protest: Will They Be Peaceful or Not?


Recent protests in the United States have resulted in the destruction of property, police brutality, violence between protesters, and, in some cases, death. An existing body of literature has successfully forecasted the time and location of protests; however, few attempts have been made to predict the nature of the demonstration. This paper analyzes news articles four days before a protest occurs in an effort to predict whether the gathering will resolve peacefully or not. I identify that sentiment scores provided the most valuable insight regarding the outcome of a protest. The findings of this study have significant implications for law enforcement agencies and property owners, allowing them to take early action to prevent dangerous situations and the destruction of property.


## Author

Mason Ricci  [@riccimason99](https://github.com/riccimason99)


## Code

All code in this repository was run using Python 3.11.4 on August 15th 2025


#### Sentiment Analysis and Classification.py

Extracts sentiments from tokenised news articles and uses them as features in several machine learning classification models.

#### Word Embd and Classifier.py

Transforms tokenised articles into weighted word embeddings and document embeddings, then uses each as a feature in several machine learning classification models.

#### Clean_and_Tokenize Text.py

This code is included to display text pre-processing and tokenisation steps; however, you will not be able to run this code as the original articles have not been included in the repository.

#### Dissertation_Final.docx
Please refer to the complete article for more detailed information regarding decisions made in the coding process.
## Data


#### ACLED
Information Reguarding Protest Time Location and if it was Peacful or not is derived from the Armed Conflict Location and Event Data Project ([ACLED](https://acleddata.com/)). ACLED data used for this project can be found in the "ACLED Protest Data" folder in the repository.

#### Text
All text was scraped from the LexisNexis News and Business [website](https://www.lexisnexis.com/en-us/products/digital-library.page?srsltid=AfmBOopIkv8qttxk9SS5XoLUhA7U6EkJwk0DBH66-Yy3DUmfMGW4XME5). Relevant articles were identified by keyword search and a date window extending four days before the protest occurred. Due to copyright issues, I was unable to make the full articles available. However, cleaned and tokenized versions of the articles are available in the CSV file named Project_DF.csv.

#### Project_DF.csv
This is the data frame used for most of the code files in the project. 

- "Binary" tells if the protest was peaceful or not. 0 for peaceful, 1 for non-peaceful. ]

- "tokens" are tokenised news articles. 

- "neg","compounds", "pos", "neu" are sentiment scores

- "untokenized" are all tokens together as one string 
