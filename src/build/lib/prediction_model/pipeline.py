# prediction_model/pipeline.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import joblib
from processing.data_handling import load_data, preprocess_data
from src.build.lib.prediction_model.config.config import COUNT_VECTORIZER_PATH, SIMILARITY_MATRIX_PATH
import os


def ensure_directory(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def preprocess_and_vectorize():
    # Load and preprocess data
    movies = load_data()
    new = preprocess_data(movies)
    
    # Vectorize text data
    count_vectorizer = CountVectorizer(stop_words='english')
    count_matrix = count_vectorizer.fit_transform(new['tags'])
    
    # Save vectorizer
    ensure_directory(COUNT_VECTORIZER_PATH)
    joblib.dump(count_vectorizer, COUNT_VECTORIZER_PATH)
    

    
    return count_matrix

def train_model():
    count_matrix = preprocess_and_vectorize()
    
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(count_matrix)
    
    # Save similarity matrix
    ensure_directory(SIMILARITY_MATRIX_PATH)
    joblib.dump(similarity_matrix, SIMILARITY_MATRIX_PATH)

if __name__ == "__main__":
    train_model()
