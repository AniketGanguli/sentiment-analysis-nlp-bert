
# Sentiment Analysis using NLP & BERT

A complete **Sentiment Analysis** project that classifies text into **Positive, Negative, or Neutral** sentiments using two approaches:
1. Traditional NLP with **TF-IDF + Machine Learning**
2. Advanced **BERT (Transformer-based) model**

The project also includes an interactive **Streamlit web application** for real-time sentiment prediction.

---

## 🔍 Project Overview

- Built end-to-end sentiment analysis pipeline
- Implemented classical ML and deep learning (BERT) approaches
- Deployed models using Streamlit for easy user interaction
- Evaluated model performance using confusion matrix

---

## 🧠 Models Used

- **TF-IDF + Machine Learning**
  - Vectorizer: TF-IDF
  - Model: Pre-trained ML classifier (stored as `.pkl`)

- **BERT (Bidirectional Encoder Representations from Transformers)**
  - Pre-trained transformer model
  - Fine-tuned for sentiment classification

---

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Transformers (Hugging Face)
- PyTorch
- Streamlit
- NLP

---

## 📁 Project Structure

```
sentiment-project/
│
├── app.py                  # Streamlit app (TF-IDF + ML)
├── app_bert.py             # Streamlit app (BERT-based)
├── main.py                 # ML model training / prediction script
├── bert_test.py            # BERT testing script
│
├── sentiment_model.pkl     # Trained ML model
├── tfidf_vectorizer.pkl    # TF-IDF vectorizer
├── Twitter_Data.csv        # Dataset
├── confusion_matrix.png    # Model evaluation output
│
├── .gitignore
└── README.md
```

---

## 🚀 How to Run the Project

### 1️⃣ Clone the repository
```bash
git clone https://github.com/AniketGanguli/sentiment-analysis-nlp-bert.git
cd sentiment-analysis-nlp-bert
```

### 2️⃣ Create & activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install streamlit pandas numpy scikit-learn torch transformers
```

### 4️⃣ Run Streamlit App

#### ▶ TF-IDF + ML App
```bash
streamlit run app.py
```

#### ▶ BERT-based App
```bash
streamlit run app_bert.py
```

Open browser at:
```
http://localhost:8501
```

---

## 📊 Output

- Real-time sentiment prediction
- Visual performance analysis using confusion matrix

---

## 📌 Use Cases

- Social media sentiment analysis
- Product review analysis
- Customer feedback classification
- NLP learning & experimentation

---

## 👤 Author

**Aniket Ganguli**  
GitHub: https://github.com/AniketGanguli

---

## ⭐ If you like this project
Give it a ⭐ on GitHub — it really helps!
