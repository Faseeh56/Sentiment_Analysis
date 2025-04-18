import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load the saved model and vectorizer
model = pickle.load(open('sentiment_model.pkl', 'rb'))  # Adjust the file name if necessary
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))  # Adjust the file name if necessary

# Define a function to predict sentiment
def predict_sentiment(review):
    # Transform the review using the fitted vectorizer
    review_transformed = vectorizer.transform([review])
    
    # Predict sentiment
    prediction = model.predict(review_transformed)[0]
    
    # Return the result
    return "Positive Review" if prediction == 1 else "Negative Review"

# Streamlit Interface
st.title("Movie Review Sentiment Analysis")

# User input for review
review = st.text_area("Enter your movie review:")

if st.button("Predict Sentiment"):
    if review:
        sentiment = predict_sentiment(review)
        st.write(f"The sentiment of the review is: **{sentiment}**")
    else:
        st.write("Please enter a review to predict sentiment.")
