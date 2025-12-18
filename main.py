import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib # For saving the model and vectorizer

# --- 1. Data Loading and Initial Cleaning ---
try:
    df = pd.read_csv('Twitter_Data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Twitter_Data.csv not found. Please ensure the file is in the correct directory.")
    exit()

# Drop rows with missing values and ensure correct data types
df.dropna(subset=['clean_text'], inplace=True)
df.dropna(subset=['category'], inplace=True)
df['category'] = df['category'].astype(int)

# --- 2. Text Preprocessing ---
# Initialize lemmatizer and stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and preprocesses a single text entry.
    - Removes punctuation
    - Converts to lowercase
    - Splits into tokens
    - Lemmatizes tokens and removes stopwords
    """
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

print("\nStarting text preprocessing...")
# Apply the function to the 'clean_text' column
df['processed_text'] = df['clean_text'].apply(preprocess_text)
print("Text preprocessing complete.")

# --- 3. Splitting Data and Vectorization ---
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_text'],
    df['category'],
    test_size=0.2, # 20% for testing
    random_state=42, # Ensures reproducible results
    stratify=df['category'] # Maintains sentiment proportion in train/test sets
)

print("\nVectorizing text data with TF-IDF...")
# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000) # Limit to the top 5000 words

# Fit on training data and transform both training and testing data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("Vectorization complete.")

# --- 4. Model Training ---
print("\nTraining the Linear SVM model...")
# Initialize and train the Linear Support Vector Machine model
model = LinearSVC(random_state=42, dual=False, max_iter=1000)
model.fit(X_train_vec, y_train)
print("Model training complete.")

# --- 5. Model Evaluation ---
print("\nEvaluating the model...")
# Make predictions on the test set
y_pred = model.predict(X_test_vec)

# Print a detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive']))

# --- 6. Confusion Matrix Visualization ---
# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Sentiment')
plt.xlabel('Predicted Sentiment')
plt.savefig('confusion_matrix.png')
print("\nSaved confusion matrix as confusion_matrix.png")

# --- 7. Save the Model and Vectorizer ---
# Save the TF-IDF vectorizer to a file
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Save the trained model to a file
joblib.dump(model, 'sentiment_model.pkl')

print("\nModel and vectorizer have been saved to disk.")
print("\nScript finished successfully!")
