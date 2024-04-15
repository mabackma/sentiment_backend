from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import cast
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from dotenv import load_dotenv
import os
import jwt
import datetime

def change_to_numeric(rating):
    try:
        return float(rating)
    except ValueError:
        return None


# Trains the model
def train_model(reviews):
    df_restaurant_reviews = pd.read_csv(reviews)

    # drop last column
    df_restaurant_reviews.drop(columns=['7514'], inplace=True)

    # Make sure that the Rating column is numeric and drop None values
    df_restaurant_reviews['Rating'] = df_restaurant_reviews['Rating'].apply(change_to_numeric)
    df_restaurant_reviews = df_restaurant_reviews[df_restaurant_reviews['Rating'] != 0]
    df_restaurant_reviews.dropna(inplace=True)

    # Add a label to column based on the score.
    df_restaurant_reviews['label'] = df_restaurant_reviews['Rating'].apply(
        lambda x: 'neu' if x == 3 else ('pos' if x >= 4 else 'neg'))

    # Drop extra columns.
    df_reviews = df_restaurant_reviews.drop(
        columns=['Restaurant', 'Reviewer', 'Rating', 'Metadata', 'Time', 'Pictures'])

    # split dataset so that x contains the review texts and y contains the sentiment labels
    x = df_reviews['Review']
    y = df_reviews['label']

    # 70% of data is used for training and 30% of data is used for testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # TF-IDF Vectorizer converts text into numerical data
    # and then trains a LinearSVC model on that data.
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LinearSVC())
    ])

    return text_clf.fit(x_train, y_train)


# Predicts the sentiment
def predict_sentiment(model, review):
    # Test the model with review
    sentiment = model.predict([review])
    if sentiment[0] == 'neg':
        return 'Negative'
    if sentiment[0] == 'pos':
        return 'Positive'
    if sentiment[0] == 'neu':
        return 'Neutral'


# Train the sentiment model with restaurant reviews
pipeline = cast(Pipeline, train_model('Restaurant_reviews.csv'))

app = Flask(__name__)
CORS(app)

load_dotenv()
# Access environment variables
env_username = os.getenv('USER_NAME')
env_password = os.getenv('PASSWORD')
secret_key = os.getenv('SECRET_KEY')


# Return JWT token
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data['username']
    password = data['password']

    if not username or not password:
        return jsonify({'message': 'Invalid username or password'}), 401

    if username == env_username and password == env_password:
        # Generate JWT token
        token = jwt.encode({'username': username, 'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)},
                           secret_key)
        return jsonify({'token': token}), 200

    return jsonify({'message': 'Invalid username or password'}), 401


# Return sentiment prediction from post
@app.route('/sentiment', methods=['POST'])
def sentiment():
    data = request.get_json()
    prediction = predict_sentiment(pipeline, data['text'])

    return {'sentiment': prediction}


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)