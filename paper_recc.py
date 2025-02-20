# %% Import all libraries

import pandas as pd
import matplotlib as plt
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import sys

#%% Filter Kaggle data

def filter_kaggle_data(path, new_file):
    """
    Filters the first 500 rows of a Kaggle dataset and saves it to a new file.
    Args:
        path (str): The file path to the original Kaggle dataset (CSV format).
        new_file (str): The file path where the filtered dataset will be saved (CSV format).
    Returns:
        None
    """
    df = pd.read_csv(f"{path}")
    df = df[:500]
    df.to_csv(f"{new_file}")

# Only run if needed
# filter_kaggle_data(r"archive\arxiv_data_210930-054931.csv", "filtered_arxiv_data.csv")
# %% Load and preprocess dataframe

def dataframe_loading_and_processing(data_path):
    """
    Load and preprocess the dataset.

    Parameters:
    data_path (str): The path to the CSV file containing the dataset.

    Returns:
    pd.DataFrame: A pandas DataFrame with unnecessary columns removed and 
    missing values filled with spaces.
    """
    df = pd.read_csv(f"{data_path}")
    df = df.drop(columns='Unnamed: 0')
    df = df.fillna(' ')
    return df

#%% Define lemmatization and stemming functions

def lemmatization_and_stemming():
    """
    Downloads necessary NLTK data and initializes lemmatizer and stemmer.

    Returns:
    tuple: A tuple containing the lemmatizer and stemmer objects.
    """
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    return lemmatizer, stemmer

# %% Transform text

def text_transform(text, lemmatizer, stemmer):
    """
    Transforms the input text by applying the following steps:
    1. Converts all text to lowercase.
    2. Removes any URLs.
    3. Removes non-word and non-whitespace characters.
    4. Removes stopwords.
    5. Applies lemmatization.
    6. Applies stemming.

    Parameters:
    text (str): The input text to be transformed.
    lemmatizer (WordNetLemmatizer): The lemmatizer object.
    stemmer (PorterStemmer): The stemmer object.

    Returns:
    str: The transformed text.
    """
    # Converting all text to lowercase
    text = text.lower()
    # Remove any URLs
    text = re.compile(r'https?://\S+').sub('', text)
    # Remove non-word and non-whitespace characters
    text = re.sub(r'[^\w\s]', '', text)
    # Stopword removal
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    # Lemmatization
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    # Stemming
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text

#%% Get top recommendations based on input

def get_recommendations(input, top_scores, data_path="filtered_arxiv_data.csv"):
    """
    Generates a list of top recommendations based on the cosine similarity 
    between the input text and the abstracts in the dataset.

    Parameters:
    data_path (str): The path to the CSV file containing the dataset.
    input (str): The input text for which recommendations are to be generated.
    top_scores (int): The number of top recommendations to return.

    Returns:
    list: A list of titles corresponding to the top recommendations.
    """
    # Reading the data
    initial_df = dataframe_loading_and_processing(data_path)
    lemmatizer, stemmer = lemmatization_and_stemming()
    df = initial_df.copy()

    # Transform column in data and input
    df['abstracts'] = df['abstracts'].apply(lambda text: text_transform(text, lemmatizer, stemmer))
    abstracts = df['abstracts'].tolist()
    abstracts.append(text_transform(input, lemmatizer, stemmer))

    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(abstracts)

    # Cosine Similarity
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    sim_scores = list(enumerate(cosine_sim[0]))
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse=True)
    
    # Return best matches
    best_indeces = [score[0] for score in sim_scores[:top_scores]]
    data = {'Title': [initial_df['titles'][index] for index in best_indeces], 'Score': [f"{round(score[1]*100, 2)}%" for score in sim_scores[:top_scores]], 'Abstract': [initial_df['abstracts'][index] for index in best_indeces]}
    return pd.DataFrame(data)

# %% Running from Terminal

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: py paper_recc.py <input> <num_scores>")
    if type(sys.argv[2]) != int:
        print("Please provide a number for the top scores.")
    if type(sys.argv[1]) != str:
        print("Please enter a valid input string")
    else:
        print(get_recommendations(sys.argv[1], int(sys.argv[2])))
