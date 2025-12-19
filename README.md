# -Movie-Recommendation-System-using-Machine-Learning
This project implements a content-based Movie Recommendation System using Natural Language Processing (NLP) and Machine Learning techniques to recommend similar movies based on user preferences.  

The system analyzes movie metadata such as genres, keywords, tagline, cast, and director, converts text data into numerical vectors using TF-IDF, and computes similarity using Cosine Similarity to generate accurate movie recommendations.

🔍 Project Workflow

Data Collection & Preprocessing

Loaded movie dataset containing 4,803 movies

Handled missing values

Selected meaningful textual features

Feature Engineering

Combined genres, keywords, tagline, cast, and director

Converted text data into numerical vectors using TF-IDF

Similarity Computation

Applied cosine similarity to measure closeness between movies

Built a similarity matrix (4803 × 4803)

Recommendation Engine

Used fuzzy matching (difflib) to handle user input errors

Retrieved top 30 most similar movies based on similarity score

🎯 Key Features

Content-based filtering

NLP-driven recommendations

Handles partial & misspelled movie names

Fast and scalable similarity computation

Interactive command-line recommendation system

🛠️ Tech Stack

Python

Pandas & NumPy

Scikit-learn

TF-IDF Vectorizer

Cosine Similarity

NLP & Machine Learning

📂 Files Included

movie_recommendation.py – main ML script

movies.csv – dataset

Day 19 Project.pdf – detailed workflow

project_demo.mp4 – project explanation video

⭐ This project demonstrates strong fundamentals in Machine Learning, NLP, and Recommendation Systems, relevant for Data Scientist / ML Engineer / Data Analyst roles.
