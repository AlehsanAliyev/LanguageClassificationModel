from datasets import load_dataset
import os

# Output directory
os.makedirs("data/ru", exist_ok=True)

def save_text_only(dataset_name, text_field, file_prefix, max_items=None):
    print(f"📥 Loading {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train")

    texts = [item[text_field] for item in dataset if item.get(text_field)]
    if max_items:
        texts = texts[:max_items]

    with open(f"data/ru/{file_prefix}.txt", "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line.strip() + "\n")

    print(f"✅ Saved {len(texts)} entries to data/ru/{file_prefix}.txt")

# # 1. Librispeech (text)
# save_text_only(
#     dataset_name="FacelessLake/noise-augmented-russian-librispeech",
#     text_field="text",
#     file_prefix="librispeech_text", # small and fast
# )
# #
# # 2. Bank Reviews (review)
# save_text_only(
#     dataset_name="Romjiik/Russian_bank_reviews",
#     text_field="review",
#     file_prefix="bank_reviews"
# )

# 3. Russian Cleared Wikipedia (text)
save_text_only(
    dataset_name="Den4ikAI/russian_cleared_wikipedia",
    text_field="sample",
    file_prefix="russian_cleared_wikipedia"
)

print("🎉 All datasets loaded and saved as plain text.")
