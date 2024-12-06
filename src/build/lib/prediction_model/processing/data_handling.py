# prediction_model/processing/data_handling.py

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
import json
from src.build.lib.prediction_model.config.config import RAW_MOVIES_DATA_PATH, RAW_CREDITS_DATA_PATH
import joblib
# Uncomment to download necessary NLTK resources if not already downloaded
# nltk.download('punkt')
# nltk.download('stopwords')

stemmer = PorterStemmer()

def load_data():
    # Load raw data
    movies = pd.read_csv(RAW_MOVIES_DATA_PATH)
    credits = pd.read_csv(RAW_CREDITS_DATA_PATH)
    
    # Merge datasets
    movies = movies.merge(credits, on='title')
    
    return movies

def preprocess_data(movies):
    # Extract relevant features
    movies = movies[['genres', 'id', 'keywords', 'overview', 'title', 'cast', 'crew']].copy()  # Ensure we are working on a copy
    
    # Clean and preprocess features
    movies['genres'] = movies['genres'].apply(lambda x: json.loads(x))
    movies['genres'] = movies['genres'].apply(lambda x: [genre['name'] for genre in x])

    movies['keywords'] = movies['keywords'].apply(lambda x: json.loads(x))
    movies['keywords'] = movies['keywords'].apply(lambda x: [keywords['name'] for keywords in x])

    movies['cast'] = movies['cast'].apply(lambda x: json.loads(x))
    movies['cast'] = movies['cast'].apply(lambda x: [character['character'] for character in x][:3])  # Limit to top 3 characters

    def get_directors(crew_list):
        return [person['name'] for person in crew_list if person.get('job') == 'Director']

    movies['crew'] = movies['crew'].apply(lambda x: json.loads(x))
    movies['directors'] = movies['crew'].apply(get_directors)

    # Tokenize and stem text data in 'overview'
    movies['overview'] = movies['overview'].fillna('')  # Handle NaN values
    movies['overview'] = movies['overview'].apply(tokenize_and_stem)
    
    # Combine features into 'tags'
    movies['tags'] = movies.apply(lambda row: ' '.join(row['genres']) + ' ' +
                                             ' '.join(row['keywords']) + ' ' +
                                             ' '.join(row['cast']) + ' ' +
                                             ' '.join(row['directors']), axis=1)
    
    # Select relevant columns
    new = movies[['title', 'tags']]
    joblib.dump(new, r'src\build\lib\prediction_model\trained_models\movie_list.pkl')
    return new

def tokenize_and_stem(text):
    tokens = word_tokenize(text)
    stemmed = [stemmer.stem(token.lower()) for token in tokens if token.isalpha()]  # Stem only alphabetical tokens
    return ' '.join(stemmed)
