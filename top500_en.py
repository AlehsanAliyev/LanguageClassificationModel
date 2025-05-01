import os
import re
from collections import Counter

# Input files and output directory
input_files = {
    "customer_support": "data/en/customer_support_inputs.txt",
    "wikipedia": "data/en/wikipedia_paragraphs.txt"
}
output_dir = "data/en/top500_words"
os.makedirs(output_dir, exist_ok=True)

# Simple tokenizer
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# Process each file
for name, path in input_files.items():
    word_counter = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            words = tokenize(line)
            word_counter.update(words)

    # Top 500 words
    top_words = word_counter.most_common(500)

    # Save
    out_path = os.path.join(output_dir, f"{name}_top500_words.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for word, freq in top_words:
            f.write(f"{word}\t{freq}\n")

print("✅ Top 500 words saved to 'data/en/top500_words/'")
