# prediction_model/predict.py

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from processing.data_handling import load_data, preprocess_data, tokenize_and_stem
from src.build.lib.prediction_model.config.config import COUNT_VECTORIZER_PATH, SIMILARITY_MATRIX_PATH
import joblib

def load_model():
    count_vectorizer = joblib.load(COUNT_VECTORIZER_PATH)
    similarity_matrix = joblib.load(SIMILARITY_MATRIX_PATH)
    return count_vectorizer, similarity_matrix

def recommend(movie_title):
    count_vectorizer, similarity_matrix = load_model()
    movies = load_data()
    
    
    index = new[new['title'] == movie_title].index[0]
    similarity_scores = list(enumerate(similarity_matrix[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    top_similar_movies = similarity_scores[1:11]
    
    recommended_movies = [new.iloc[i[0]]['title'] for i in top_similar_movies]
    
    return top_similar_movies

if __name__ == "__main__":
    new = joblib.load(r'src\build\lib\prediction_model\trained_models\movie_list.pkl')
    movie_title = "Pirates of the Caribbean: Dead Man's Chest"
    recommended_movies = recommend(movie_title)
    print(f"Recommended movies for '{movie_title}':")
    # for movie in recommended_movies:
    #     print(movie)

    for i, score in recommended_movies:
        print(f"{new.iloc[i]['title']} (Similarity Score: {score:.2f})")
