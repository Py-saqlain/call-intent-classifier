import pandas as pd

pd.set_option('display.max_colwidth', None)

df = pd.read_csv("data/calls.csv")

# 1. Class distribution — most important check
print("Class distribution:\n", df['label'].value_counts())
print("\nClass distribution (%):\n", df['label'].value_counts(normalize=True) * 100)

# 2. Check for missing/null values
print("\nMissing values:\n", df.isnull().sum())

# 3. Check for duplicate rows
print("\nDuplicate rows:", df.duplicated().sum())

# 4. Text length stats (word count) — overall and per class
df['word_count'] = df['text'].apply(lambda x: len(x.split()))
print("\nOverall word count stats:\n", df['word_count'].describe())
print("\nWord count by label:\n", df.groupby('label')['word_count'].describe())

# 5. Most common words per class (quick signal check)
from collections import Counter
import re

for label in df['label'].unique():
    text = " ".join(df[df['label'] == label]['text']).lower()
    words = re.findall(r'\b\w+\b', text)
    common = Counter(words).most_common(10)
    print(f"\nTop words in '{label}':", common)