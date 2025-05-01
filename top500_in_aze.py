import os
import re
from collections import Counter

# Define input files and output directory
input_files = {
    "azbanks": "data/az/azbanks.txt",
    "anl-news": "data/az/anl-news.txt"
}
output_dir = "data/az/top500_words"
os.makedirs(output_dir, exist_ok=True)


# Basic tokenizer (you can later replace with a language-specific one if needed)
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# Process each file
for name, path in input_files.items():
    word_counter = Counter()

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            words = tokenize(line)
            word_counter.update(words)

    # Get top 500 words
    top_words = word_counter.most_common(500)

    # Save to file
    output_path = os.path.join(output_dir, f"{name}_top500_words.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for word, freq in top_words:
            f.write(f"{word}\t{freq}\n")

print("✅ Top 500 words with frequencies saved in 'data/az/top500_words/'")
