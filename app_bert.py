import streamlit as st
from transformers import pipeline
import pandas as pd

# --- Page Config ---
st.set_page_config(page_title="Advanced Sentiment Analyzer", page_icon="🤖", layout="centered")

# --- Model Loading ---
@st.cache_resource
def load_sentiment_model():
    """Loads the sentiment analysis model from Hugging Face."""
    model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
    return sentiment_pipeline

st.title("Advanced Sentiment Analysis 🤖")
st.write("This app uses a RoBERTa model fine-tuned on millions of tweets to predict sentiment.")

# Load the model (with a spinner to show progress)
with st.spinner("Loading the advanced model... This may take a moment."):
    sentiment_pipeline = load_sentiment_model()

st.success("Model loaded successfully!")

# --- User Input ---
st.header("Analyze Your Own Text")
user_input = st.text_area("Enter a sentence here:", "The government's new policy is a fantastic step forward!")

if st.button("Analyze Sentiment"):
    if user_input:
        # The model returns a list with a single dictionary
        result = sentiment_pipeline(user_input)[0]
        label = result['label'].capitalize()
        score = result['score']

        # Display the result with a nice color-coded box
        if label == 'Positive':
            st.success(f"**Predicted Sentiment: {label}** (Confidence: {score:.2f})")
        elif label == 'Negative':
            st.error(f"**Predicted Sentiment: {label}** (Confidence: {score:.2f})")
        else:
            st.info(f"**Predicted Sentiment: {label}** (Confidence: {score:.2f})")
    else:
        st.warning("Please enter a sentence to analyze.")

