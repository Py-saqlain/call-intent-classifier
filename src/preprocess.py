import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os

df = pd.read_csv("data/calls.csv")

# 1. Basic text cleaning
df['text'] = df['text'].str.lower().str.strip()

# 2. Train/test split — BEFORE vectorizing (avoid data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']   # keeps class balance same in train & test
)

# 3. TF-IDF vectorization
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),      # unigrams + bigrams (e.g. "cancel", "cancel my")
    stop_words='english',
    max_features=3000
)

X_train_vec = vectorizer.fit_transform(X_train)   # fit only on train
X_test_vec = vectorizer.transform(X_test)          # transform test using same vocab

# 4. Save everything for the next step (train.py)
os.makedirs("models", exist_ok=True)
with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("data/splits.pkl", "wb") as f:
    pickle.dump((X_train_vec, X_test_vec, y_train, y_test), f)

print("Train shape:", X_train_vec.shape)
print("Test shape:", X_test_vec.shape)
print("Vocabulary size:", len(vectorizer.vocabulary_))