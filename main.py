import streamlit as st
import streamlit_option_menu
from streamlit_extras.stoggle import stoggle
from processing import preprocess
from processing.display import Main
import requests
import pandas as pd
import os
import joblib
from PIL import Image

# Setting the wide mode as default
st.set_page_config(layout="wide")

displayed = []

if 'movie_number' not in st.session_state:
    st.session_state['movie_number'] = 0

if 'selected_movie_name' not in st.session_state:
    st.session_state['selected_movie_name'] = ""

if 'user_menu' not in st.session_state:
    st.session_state['user_menu'] = ""

# Ensure directory
def ensure_directory(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Load data from the pickle file
ensure_directory(r'src/build/lib/prediction_model/trained_models/list.pkl')
data = joblib.load(r'src/build/lib/prediction_model/trained_models/list.pkl')

# Load the sentiment analysis model
ensure_directory(r'src/build/lib/prediction_model/trained_models/sentiment_model.pkl')
with open(r'src/build/lib/prediction_model/trained_models/sentiment_model.pkl', 'rb') as model_file:
    tfidf_vectorizer, naive_bayes = joblib.load(model_file)

# Function to fetch movie poster from TMDb
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=f772f057340a7021d5fc62995e6a3f97&language=en-US"
        data = requests.get(url).json()
        poster_path = data['poster_path']
        full_path = "https://image.tmdb.org/t/p/w500/" + poster_path
        return full_path
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Function to get movie reviews from TMDb API
def get_movie_reviews(movie_id):
    url = f'https://api.themoviedb.org/3/movie/{movie_id}/reviews'
    params = {'api_key': 'f772f057340a7021d5fc62995e6a3f97'}

    response = requests.get(url, params=params)
    data = response.json()

    return data.get('results', [])

# Function to predict sentiment using the loaded model
def predict_sentiment(review):
    transformed_review = tfidf_vectorizer.transform([review])
    prediction = naive_bayes.predict(transformed_review)
    return prediction[0]

# Assuming the data is a list of dictionaries
df = pd.DataFrame(data)

def main():
    def initial_options():
        # To display menu
        st.session_state.user_menu = streamlit_option_menu.option_menu(
            menu_title='What are you looking for? 👀',
            options=['Recommend me a similar movie', 'Describe me a movie', 'Check all Movies', 'Sentiment Analysis'],
            icons=['film', 'film', 'film', 'film'],
            menu_icon='list',
            orientation="horizontal",
        )

        if st.session_state.user_menu == 'Recommend me a similar movie':
            recommend_display()

        elif st.session_state.user_menu == 'Describe me a movie':
            display_movie_details()

        elif st.session_state.user_menu == 'Check all Movies':
            paging_movies()

        elif st.session_state.user_menu == 'Sentiment Analysis':
            sentiment_analysis()

    def recommend_display():
        st.title('Movie Recommender System')

        selected_movie_name = st.selectbox(
            'Select a Movie...', new_df['title'].values
        )

        rec_button = st.button('Recommend')
        if rec_button:
            st.session_state.selected_movie_name = selected_movie_name
            recommendation_tags(new_df, selected_movie_name, r'Files/similarity_tags_tags.pkl', "are")
            recommendation_tags(new_df, selected_movie_name, r'Files/similarity_tags_genres.pkl', "on the basis of genres are")
            recommendation_tags(new_df, selected_movie_name, r'Files/similarity_tags_tprduction_comp.pkl', "from the same production company are")
            recommendation_tags(new_df, selected_movie_name, r'Files/similarity_tags_keywords.pkl', "on the basis of keywords are")
            recommendation_tags(new_df, selected_movie_name, r'Files/similarity_tags_tcast.pkl', "on the basis of cast are")

    def recommendation_tags(new_df, selected_movie_name, pickle_file_path, str):
        movies, posters = preprocess.recommend(new_df, selected_movie_name, pickle_file_path)
        st.subheader(f'Best Recommendations {str}...')

        rec_movies = []
        rec_posters = []
        cnt = 0
        for i, j in enumerate(movies):
            if cnt == 5:
                break
            if j not in displayed:
                rec_movies.append(j)
                rec_posters.append(posters[i])
                displayed.append(j)
                cnt += 1

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(rec_movies[0])
            st.image(rec_posters[0])
        with col2:
            st.text(rec_movies[1])
            st.image(rec_posters[1])
        with col3:
            st.text(rec_movies[2])
            st.image(rec_posters[2])
        with col4:
            st.text(rec_movies[3])
            st.image(rec_posters[3])
        with col5:
            st.text(rec_movies[4])
            st.image(rec_posters[4])

    def display_movie_details():
        selected_movie_name = st.session_state.selected_movie_name
        info = preprocess.get_details(selected_movie_name)

        with st.container():
            image_col, text_col = st.columns((1, 2))
            with image_col:
                st.text('\n')
                st.image(info[0])

            with text_col:
                st.text('\n')
                st.text('\n')
                st.title(selected_movie_name)
                st.text('\n')
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.text("Rating")
                    st.write(info[8])
                with col2:
                    st.text("No. of ratings")
                    st.write(info[9])
                with col3:
                    st.text("Runtime")
                    st.write(info[6])

                st.text('\n')
                st.write("Overview")
                st.write(info[3], wrapText=False)
                st.text('\n')
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.text("Release Date")
                    st.text(info[4])
                with col2:
                    st.text("Budget")
                    st.text(info[1])
                with col3:
                    st.text("Revenue")
                    st.text(info[5])

                st.text('\n')
                col1, col2, col3 = st.columns(3)
                with col1:
                    genre_str = " . ".join(info[2])
                    st.text("Genres")
                    st.write(genre_str)

                with col2:
                    available_str = " . ".join(info[13])
                    st.text("Available in")
                    st.write(available_str)

                with col3:
                    st.text("Directed by")
                    st.text(info[12][0])
                st.text('\n')

        st.header('Cast')
        cnt = 0
        urls = []
        bio = []
        for i in info[14]:
            if cnt == 5:
                break
            url, biography = preprocess.fetch_person_details(i)
            urls.append(url)
            bio.append(biography)
            cnt += 1

        col1, col2, col3, col4, col5 = st.columns(5)
        for idx, col in enumerate([col1, col2, col3, col4, col5]):
            with col:
                st.image(urls[idx])
                stoggle("Show More", bio[idx])

    def paging_movies():
        max_pages = movies.shape[0] // 10 - 1

        col1, col2, col3 = st.columns([1, 9, 1])

        with col1:
            st.text("Previous page")
            if st.button("Prev"):
                if st.session_state['movie_number'] >= 10:
                    st.session_state['movie_number'] -= 10

        with col2:
            new_page_number = st.slider("Jump to page number", 0, max_pages, st.session_state['movie_number'] // 10)
            st.session_state['movie_number'] = new_page_number * 10

        with col3:
            st.text("Next page")
            if st.button("Next"):
                if st.session_state['movie_number'] + 10 < len(movies):
                    st.session_state['movie_number'] += 10

        display_all_movies(st.session_state['movie_number'])

    def display_all_movies(start):
        i = start
        with st.container():
            col1, col2, col3, col4, col5 = st.columns(5)
            for col in [col1, col2, col3, col4, col5]:
                id = movies.iloc[i]['movie_id']
                link = preprocess.fetch_posters(id)
                col.image(link, caption=movies['title'][i])
                i += 1

        with st.container():
            col1, col2, col3, col4, col5 = st.columns(5)
            for col in [col1, col2, col3, col4, col5]:
                id = movies.iloc[i]['movie_id']
                link = preprocess.fetch_posters(id)
                col.image(link, caption=movies['title'][i])
                i += 1

    def sentiment_analysis():
        st.title('Sentiment Analysis')
        
        selected_movie_name = st.session_state.selected_movie_name
        
        if selected_movie_name:
            selected_movie_id = df[df['title'] == selected_movie_name]['movie_id'].values[0]
            poster_path = fetch_poster(selected_movie_id)
            
            col1, col2 = st.columns(2)
            with col1:
                if poster_path:
                    st.subheader("   ")
                    img = Image.open(requests.get(poster_path, stream=True).raw)
                    st.image(img, width=300)
                else:
                    st.warning("Poster not available for this movie.")

            with col2:
                st.title(f" {selected_movie_name}")
                st.subheader("Overview")
                st.write(df[df['movie_id'] == selected_movie_id]['overview'].values[0])

                # col5, col6 = st.columns(2)
                # with col5:
                #     st.subheader("Genre")
                #     genres_list = df[df['movie_id'] == selected_movie_id]['genres'].values[0]
                #     st.write(', '.join(genres_list))

                # with col6:
                #     st.subheader("Keywords")
                #     keywords_list = df[df['movie_id'] == selected_movie_id]['keywords'].values[0]
                #     keywords_list = keywords_list[:3]
                #     st.write(', '.join(keywords_list))

                # col3, col4 = st.columns(2)
                # with col3:
                #     st.subheader("Cast")
                #     cast_list = df[df['movie_id'] == selected_movie_id]['cast'].values[0]
                #     st.write(' '.join(cast_list))
                # with col4:
                #     st.subheader("Director")
                #     director_list = df[df['movie_id'] == selected_movie_id]['crew'].values[0]
                #     st.write(' '.join(director_list))

            reviews = get_movie_reviews(selected_movie_id)
            reviews = reviews[:4]
            if reviews:
                st.subheader("Reviews and Sentiment Analysis:")
                for review in reviews:
                    full_content = review['content']
                    author = review['author']
                    sentiment_prediction = predict_sentiment(full_content)
                    result_text = "Positive" if sentiment_prediction == 1 else "Negative"
                    color = "#B3FFAE" if sentiment_prediction == 1 else "#FF6464"
                    text_color = "black" if sentiment_prediction == 1 else "white"

                    with st.expander(f"Review by {author}"):
                        st.markdown(
                            f"""
                            <div style="background-color:{color}; padding: 10px; border-radius: 5px; color:{text_color}">
                                <p style="font-size: 18px;"><b>Full Review:</b> {full_content}</p>
                                <p style="font-size: 18px;"><b>Sentiment:</b> {result_text}</p>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
            else:
                st.warning("No reviews found for the selected movie.")
        else:
            st.warning("No movie selected. Please select a movie from the 'Recommend me a similar movie' tab.")


    # Main execution starts here
    with Main() as bot: 
        bot.main_() 
        new_df, movies, movies2 = bot.getter() 
        initial_options()


if __name__ == '__main__':
    main()