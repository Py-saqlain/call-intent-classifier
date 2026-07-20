import pandas as pd
import os
import sys

def quick_label(validation_csv, data_folder):
    df = pd.read_csv(validation_csv)
    df['true_label'] = df['true_label'].astype(object)  # force text column, not float

    label_map = {"b": "booking", "t": "transfer", "c": "cancellation", "m": "modification", "s": "SKIP"}
    

    for i, row in df.iterrows():
        if pd.notna(row['true_label']) and row['true_label'] != "":
            continue  # already labeled, skip

        filepath = os.path.join(data_folder, row['filename'])
        if not os.path.exists(filepath):
            print(f"Missing file: {row['filename']}")
            continue

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"[{i+1}/{len(df)}] {row['filename']}\n")
        print(text)
        print("\n" + "="*60)
        choice = input("Label: (b)ooking (t)ransfer (c)ancellation (m)odification (s)kip (q)uit: ").strip().lower()

        if choice == "q":
            break
        if choice in label_map and choice != "s":
            df.at[i, 'true_label'] = label_map[choice]
            df.to_csv(validation_csv, index=False)  # save after every label, don't lose progress

    print("Done. Saved to", validation_csv)

if __name__ == "__main__":
    quick_label(sys.argv[1], sys.argv[2])