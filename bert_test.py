import pandas as pd
from transformers import pipeline

print("Loading a sentiment analysis model fine-tuned on tweets (this may take a moment)...")
# This model (RoBERTa) is specifically trained on Twitter data and is better for this task.
# It also includes a 'Neutral' class.
# Model link: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest
model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_path, tokenizer=model_path)
print("Model loaded successfully.")

# --- Load your original dataset to get some test samples ---
try:
    df = pd.read_csv('Twitter_Data.csv')
    df.dropna(subset=['clean_text'], inplace=True)
except FileNotFoundError:
    print("\nError: Twitter_Data.csv not found.")
    print("Please make sure the dataset file is in the project folder.")
    exit()

# --- Select a few sample tweets to test ---
# We'll pick a few interesting examples from your data
sample_texts = [
    "when modi promised “minimum government maximum governance” expected him begin the difficult job reforming the state", # Expected Negative
    "talk all the nonsense and continue all the drama will vote for modi", # Expected Neutral/Positive
    "what did just say vote for modi welcome bjp told you rahul the main campaigner for modi", # Expected Positive
    "this is not good and will cause many problems" # A generic negative sentence
]

print("\n--- Analyzing Sample Sentences ---")
# Analyze the samples and print the results
results = sentiment_pipeline(sample_texts)

for text, result in zip(sample_texts, results):
    print(f"\nText: '{text}'")
    # The labels from this model are 'Negative', 'Neutral', 'Positive'
    print(f"Predicted Sentiment: {result['label']} (Score: {result['score']:.4f})")
