# config/config.py

# API key for external services
API_KEY = "f772f057340a7021d5fc62995e6a3f97"

# File paths
RAW_MOVIES_DATA_PATH = "https://raw.githubusercontent.com/CTopham/TophamRepo/master/Movie%20Project/Resources/tmdb_5000_movies.csv"
RAW_CREDITS_DATA_PATH = "https://raw.githubusercontent.com/CTopham/TophamRepo/master/Movie%20Project/Resources/tmdb_5000_credits.csv"

# Paths for serialized models and data
PREPROCESSED_DATA_PATH = "trained_models/preprocessed_data.pkl"
COUNT_VECTORIZER_PATH = r'src\build\lib\prediction_model\trained_models\count_vectorizer.pkl'
SIMILARITY_MATRIX_PATH = r'src\build\lib\prediction_model\trained_models\similarity_matrix.pkl'

# Paths for input and output data
TRAIN_DATA_PATH = "datasets/train.csv"
TEST_DATA_PATH = "datasets/test.csv"

# Other configurations
STOPWORDS_PATH = "preprocessing/stopwords.txt"


MOVIE_LIST_PATH=r'src\build\lib\prediction_model\trained_models\movie_list.pkl'
