import pandas as pd
import random
import csv
import sys

def create_validation_sample(predictions_csv, n, output_csv):
    df = pd.read_csv(predictions_csv)
    df = df[df['intent'] != 'ERROR']  # only sample from successfully classified rows
    sample = df.sample(min(n, len(df)), random_state=42)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_label"])
        for filename in sample['filename']:
            writer.writerow([filename, ""])
    print(f"Created {output_csv} with {len(sample)} files — fill in true_label manually")

if __name__ == "__main__":
    create_validation_sample(sys.argv[1], int(sys.argv[2]), sys.argv[3])