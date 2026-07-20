import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import os

# Load the saved splits from preprocess.py
with open("data/splits.pkl", "rb") as f:
    X_train_vec, X_test_vec, y_train, y_test = pickle.load(f)

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "svm": LinearSVC(),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
}

os.makedirs("models", exist_ok=True)

for name, model in models.items():
    model.fit(X_train_vec, y_train)
    with open(f"models/{name}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Trained and saved: {name}")

print("\nAll models trained.")