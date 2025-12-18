import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Ensure NLTK data is available ---
# This will download the necessary packages if they are not found.
try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')
try:
    # Test if wordnet is available
    from nltk.corpus import wordnet
    wordnet.ensure_loaded()
except LookupError:
    print("Downloading NLTK wordnet...")
    nltk.download('wordnet')


# --- Text Preprocessing Function (Copied from main.py) ---
# We copy this function here to make the app independent of the main.py script
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and preprocesses a single text entry.
    """
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    tokens = text.split()
    # Lemmatize and remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# --- Load the saved model and vectorizer ---
try:
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    model = joblib.load('sentiment_model.pkl')
except FileNotFoundError:
    st.error("Model or vectorizer not found. Please run main.py first to train and save them.")
    st.stop()

# --- Streamlit App Interface ---
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊", layout="centered")
st.title("Live Sentiment Analysis 💬")
st.write("Enter a sentence below to see its predicted sentiment. The model was trained on a dataset of tweets about Indian politics.")

# --- User Input ---
user_input = st.text_area("Your sentence here:", "Modi is doing a great job for India!")

# --- Prediction Logic ---
if st.button("Analyze Sentiment"):
    if user_input:
        # 1. Preprocess the user's input
        processed_input = preprocess_text(user_input)

        # 2. Vectorize the processed input using the loaded vectorizer
        vectorized_input = vectorizer.transform([processed_input])

        # 3. Predict the sentiment using the loaded model
        prediction = model.predict(vectorized_input)[0]

        # 4. Display the result with an emoji
        sentiment_map = {-1: 'Negative 😠', 0: 'Neutral 😐', 1: 'Positive 😊'}
        result = sentiment_map.get(prediction, "Unknown")
        
        if prediction == 1:
            st.success(f"**Predicted Sentiment:** {result}")
        elif prediction == -1:
            st.error(f"**Predicted Sentiment:** {result}")
        else:
            st.info(f"**Predicted Sentiment:** {result}")
    else:
        st.warning("Please enter a sentence to analyze.")
