# processing/preprocessing.py

from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
from src.build.lib.prediction_model.config.config import STOPWORDS_PATH

nltk.download('punkt')

stemmer = PorterStemmer()

def tokenize_and_stem(text):
    tokens = word_tokenize(text.lower())
    stemmed = [stemmer.stem(token) for token in tokens if token.isalpha() and token not in load_stopwords(STOPWORDS_PATH)]
    return stemmed
