from transliterate import translit
import os

# Create output folder
os.makedirs("data/ru", exist_ok=True)

def transliterate_file(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as f_in:
        lines = [line.strip() for line in f_in if line.strip()]

    transliterated_lines = [translit(line, 'ru', reversed=True) for line in lines]

    with open(output_path, "w", encoding="utf-8") as f_out:
        for line in transliterated_lines:
            f_out.write(line + "\n")

    print(f"✅ Transliterated {len(lines)} lines from {input_path} → {output_path}")

# Apply to your Russian Wikipedia and bank review files
transliterate_file("data/ru/russian_cleared_wikipedia.txt", "data/ru/russian_cleared_wikipedia_translit.txt")
# transliterate_file("data/ru/bank_reviews.txt", "data/ru/bank_reviews_translit.txt")
