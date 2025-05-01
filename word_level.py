import os
import csv

# Define input folders and their language tags
lang_dirs = {
    "az": "data/az/top500_words",
    "en": "data/en/top500_words",
    "ru": "data/ru/top_words"
}

# Output file
output_path = "data/corpus/word_corpus.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Collect words
rows = []

for lang, folder in lang_dirs.items():
    for file_name in os.listdir(folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder, file_name)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip().split("\t")[0]  # take just the word
                    if word:  # skip empty
                        rows.append((word, lang))

# Truncate to ~3000 max words (optional)
rows = rows[:3000]

# Write to CSV
with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["text", "lang"])
    writer.writerows(rows)

print(f"✅ Word-level corpus saved to {output_path} ({len(rows)} entries)")
