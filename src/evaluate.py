import pickle
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

with open("data/splits.pkl", "rb") as f:
    X_train_vec, X_test_vec, y_train, y_test = pickle.load(f)

model_names = ["logistic_regression", "svm", "random_forest", "decision_tree", "knn"]

results = {}

for name in model_names:
    with open(f"models/{name}.pkl", "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print(f"\n{'='*50}")
    print(f"MODEL: {name}")
    print(f"{'='*50}")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds, labels=model.classes_))
    print("Labels order:", model.classes_)

# Final comparison table
print(f"\n{'='*50}")
print("FINAL COMPARISON")
print(f"{'='*50}")
comparison_df = pd.DataFrame(results.items(), columns=["Model", "Accuracy"]).sort_values("Accuracy", ascending=False)
print(comparison_df.to_string(index=False))