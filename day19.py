# Movie Recommendation System (Day 19 Project)

# =========================
# Importing Dependencies
# =========================
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# Data Collection & Preprocessing
# =========================
# Load dataset (make sure Movies.csv is in the same folder)
data = pd.read_csv('Movies.csv')

# Features used for recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']

# Replace null values with empty string
for feature in selected_features:
    data[feature] = data[feature].fillna('')

# Combine selected features into one string
combined_features = (
    data['genres'] + ' ' +
    data['keywords'] + ' ' +
    data['tagline'] + ' ' +
    data['cast'] + ' ' +
    data['director']
)

# =========================
# Vectorization using TF-IDF
# =========================
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)

# =========================
# Cosine Similarity
# =========================
similarity = cosine_similarity(feature_vectors)

# =========================
# Movie Recommendation Logic
# =========================
def recommend_movies():
    movie_name = input('Enter your favourite movie name: ')

    list_of_all_titles = data['title'].tolist()

    # Find closest matching movie name
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        print('No close match found. Please try another movie.')
        return

    close_match = find_close_match[0]
    index_of_the_movie = data[data.title == close_match]['index'].values[0]

    # Similarity scores
    similarity_score = list(enumerate(similarity[index_of_the_movie]))

    # Sort movies based on similarity score
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    print('\nMovies suggested for you:\n')

    i = 1
    for movie in sorted_similar_movies:
        index = movie[0]
        title_from_index = data[data.index == index]['title'].values[0]
        if i <= 30:
            print(f"{i}. {title_from_index}")
            i += 1

# =========================
# Run the Recommendation System
# =========================
if __name__ == '__main__':
    recommend_movies()
