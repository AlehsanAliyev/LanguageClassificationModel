import os
import re
from collections import Counter

# Input files with desired top-N sizes
input_files = {
    "bank_reviews": ("data/ru/bank_reviews.txt", 270),
    "russian_wiki": ("data/ru/russian_cleared_wikipedia.txt", 270),
    "bank_reviews_translit": ("data/ru/bank_reviews_translit.txt", 153),
    "russian_wiki_translit": ("data/ru/russian_cleared_wikipedia_translit.txt", 154),
    "librispeech": ("data/ru/librispeech_text.txt", 153)
}

# Output directory
output_dir = "data/ru/top_words"
os.makedirs(output_dir, exist_ok=True)

# Tokenizer that supports Cyrillic + Latin words
def tokenize(text):
    return re.findall(r"\b[\w\d]+\b", text.lower(), re.UNICODE)

# Process each file
for name, (path, top_n) in input_files.items():
    word_counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            words = tokenize(line)
            word_counter.update(words)

    # Save top N words
    out_path = os.path.join(output_dir, f"{name}_top{top_n}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for word, freq in word_counter.most_common(top_n):
            f.write(f"{word}\t{freq}\n")

print("✅ Saved top 1000 words (split across 5 Russian files) in data/ru/top_words/")
