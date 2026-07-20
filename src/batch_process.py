import os
import glob
import csv
import time
from llm_classify import classify_transcript

def process_folder(folder_path, output_csv):
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    results = []

    for i, filepath in enumerate(files):
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        try:
            result = classify_transcript(text)
            results.append({
                "filename": os.path.basename(filepath),
                "intent": result.get("intent"),
                "confidence": result.get("confidence"),
                "reason": result.get("reason"),
            })
        except Exception as e:
            results.append({"filename": os.path.basename(filepath), "intent": "ERROR", "confidence": 0, "reason": str(e)})

        if i % 50 == 0:
            print(f"{i}/{len(files)} done")
        time.sleep(0.1)  # be gentle on rate limits

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "intent", "confidence", "reason"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Done — {len(results)} files processed → {output_csv}")

if __name__ == "__main__":
    import sys
    process_folder(sys.argv[1], sys.argv[2])