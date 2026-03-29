# BookVault Professional Hybrid Recommender System

## Overview

BookVault is a recommendation system that combines multiple machine learning techniques to provide accurate and meaningful book suggestions. It uses a hybrid approach by integrating content-based filtering, collaborative filtering, and metadata-based bias.

## Features

* Hybrid scoring engine that combines multiple data signals into a single match score
* Content similarity using TF-IDF and cosine similarity on book descriptions
* User correlation using Pearson correlation to identify similar user preferences
* Author bias to prioritize books from the same author
* Interactive user interface built using Streamlit
* Performance optimization using caching for handling large datasets
* Cold start handling using fallback to content-based recommendations

## Tech Stack

* Programming Language: Python 3.9 or higher
* Data Analysis: pandas, numpy
* Machine Learning: scikit-learn
* Frontend: Streamlit

## Dataset Structure

The system requires three CSV files:

1. listing.csv
   Contains book metadata such as book id, name, author, and genre

2. books.csv
   Contains user interaction data including user id, book id, and ratings

3. description.csv
   Contains book descriptions used for natural language processing

## Installation

Step 1: Clone the repository

git clone
cd bookvault

Step 2: Install dependencies

pip install pandas numpy scikit-learn streamlit

## Configuration

Place all required CSV files in the root directory.
The system automatically maps book ids across files to maintain consistency.

## Execution

Run the Streamlit application:

streamlit run streamlit_app.py

## How the Hybrid Logic Works

The system is built around the ProfessionalRecommender class.
It calculates a final score for each book using a weighted combination of three components:

Final Score = (Content Score * 0.4) + (Correlation Score * 0.4) + (Author Bias * 0.2)

This approach ensures that recommendations are:

* Similar in content
* Preferred by users with similar tastes
* Aligned with familiar authors

## Future Improvements

* Add deep learning-based recommendation models
* Improve personalization using user history tracking
* Deploy as a scalable web service
* Add support for multi-language datasets

## Acknowledgements

* scikit-learn for machine learning tools
* Streamlit for UI framework
* Open datasets used for training and testing

If you want, I can also:

* Convert both READMEs into a **single portfolio project format**
* Add **resume bullet points for this project**
* Help you prepare **interview explanation for this system**
