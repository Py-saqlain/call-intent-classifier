import pandas as pd
import sys

def compute_accuracy(validation_csv, predictions_csv):
    val = pd.read_csv(validation_csv)
    preds = pd.read_csv(predictions_csv)
    merged = val.merge(preds, on="filename")
    merged = merged[merged['true_label'].notna() & (merged['true_label'] != "")]

    accuracy = (merged['true_label'] == merged['intent']).mean()
    print(f"Accuracy on {len(merged)} verified samples: {accuracy:.2%}")

    mismatches = merged[merged['true_label'] != merged['intent']]
    if len(mismatches) > 0:
        print(f"\n{len(mismatches)} mismatches:")
        print(mismatches[['filename', 'true_label', 'intent']].to_string(index=False))

if __name__ == "__main__":
    compute_accuracy(sys.argv[1], sys.argv[2])