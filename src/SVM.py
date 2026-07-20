import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import pandas as pd

df = pd.read_csv("data/calls.csv")
df['text'] = df['text'].str.lower().str.strip()

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=3000)
X = vectorizer.fit_transform(df['text'])
y = df['label']

model = LinearSVC()
model.fit(X, y)

# Save final model + vectorizer
with open("models/final_svm.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/final_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Final SVM model trained on full dataset and saved.")

# --- Prediction function ---
def predict_intent(text):
    text = text.lower().strip()
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]


if __name__ == "__main__":
    while True:
        user_input = input("\nEnter a call sentence (or 'quit' to exit): ")
        if user_input.lower() == "quit":
            break
        prediction = predict_intent(user_input)
        print(f"Predicted intent: {prediction}")