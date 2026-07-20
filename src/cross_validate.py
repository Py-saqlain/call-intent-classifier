import pandas as pd
import pickle
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("data/calls.csv")
df['text'] = df['text'].str.lower().str.strip()

vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_features=3000)
X = vectorizer.fit_transform(df['text'])
y = df['label']

models = {
    "logistic_regression": LogisticRegression(max_iter=1000),
    "svm": LinearSVC(),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "decision_tree": DecisionTreeClassifier(random_state=42),
    "knn": KNeighborsClassifier(n_neighbors=5),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    results[name] = (scores.mean(), scores.std())
    print(f"{name}: mean={scores.mean():.3f}, std={scores.std():.3f}, folds={scores.round(3)}")

print("\nFinal ranking:")
for name, (mean, std) in sorted(results.items(), key=lambda x: -x[1][0]):
    print(f"{name}: {mean:.3f} (+/- {std:.3f})")