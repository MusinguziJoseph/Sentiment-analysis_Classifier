import streamlit as st
import pickle
import re
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the SVM model and TF-IDF vectorizer
with open('svm_model_tfidf.pkl', 'rb') as model_file:
    svm_model_tfidf = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Define a function to preprocess a new review
def preprocess_review(review):
    review = re.sub('[^a-zA-Z0-9]', ' ', review)
    review = re.sub('\s+', ' ', review)
    return review

# Define a function to predict sentiment
def predict_sentiment(review):
    # Preprocess the review
    review = preprocess_review(review)
    
    # Transform the review using the TF-IDF vectorizer
    review_vectorized = tfidf_vectorizer.transform([review])
    
    # Predict the sentiment using the SVM model
    sentiment = svm_model_tfidf.predict(review_vectorized)
    
    # Maping the predicted sentiment back to its label
    sentiment_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    return sentiment_label[sentiment[0]]

# Streamlit app
def main():
    st.title('Sentiment Analyzer Engine')
    st.write('This is a sentiment analysis web application.')

    # Input text box for user to enter a review
    review = st.text_area('Enter your review here:', '')

    if st.button('Predict'):
        if review:
            predicted_sentiment = predict_sentiment(review)
            st.write('Predicted Sentiment:', predicted_sentiment)
        else:
            st.write('Please enter a review.')

if __name__ == '__main__':
    main()
