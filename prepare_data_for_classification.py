import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Updated file paths
word_path = "data/corpus/word_corpus2.csv"
sentence_path = "data/corpus/sentence_corpus_balanced2.csv"
multi_path = "data/corpus/multisentence_corpus_balanced2.csv"

# Language label mapping
lang2label = {"az": 0, "en": 1, "ru": 2}

# Read and label
def load_and_label(path):
    df = pd.read_csv(path)
    df = df[df["lang"].isin(lang2label)]  # Filter valid
    df["label"] = df["lang"].map(lang2label)
    return df[["text", "label"]]

# Load datasets
df_word = load_and_label(word_path)
df_sent = load_and_label(sentence_path)
df_multi = load_and_label(multi_path)

# Combine
df_all = pd.concat([df_word, df_sent, df_multi], ignore_index=True)

# Shuffle and split
train_df, test_df = train_test_split(df_all, test_size=0.1, random_state=42, stratify=df_all["label"])

# Save
os.makedirs("data/final2", exist_ok=True)
train_df.to_csv("data/final2/train.csv", index=False)
test_df.to_csv("data/final2/test.csv", index=False)

print("✅ Data prepared.")
print(f" - Train samples: {len(train_df)}")
print(f" - Test samples:  {len(test_df)}")
