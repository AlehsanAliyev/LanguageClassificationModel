from datasets import load_dataset
import os

# Create output directory
os.makedirs("data/en", exist_ok=True)

def save_dataset_text(dataset_name, split, text_field, file_name, limit=None, trust_remote_code=False, config_name=None):
    print(f"📥 Loading {dataset_name}...")
    if config_name:
        ds = load_dataset(dataset_name, name=config_name, split=split, trust_remote_code=trust_remote_code)
    else:
        ds = load_dataset(dataset_name, split=split, trust_remote_code=trust_remote_code)

    texts = [ex[text_field].strip() for ex in ds if ex.get(text_field)]
    if limit:
        texts = texts[:limit]

    path = f"data/en/{file_name}.txt"
    with open(path, "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line + "\n")

    print(f"✅ Saved {len(texts)} lines to {path}")

#
# # 1. Wikipedia (lightweight subset) - 13,000
# save_dataset_text(
#     dataset_name="wikimedia/wikipedia",
#     split="train",
#     text_field="text",
#     file_name="wikipedia_en",
#     limit=13000,
#     trust_remote_code=True,
#     config_name="20231101.ace"
# )
#
# save_dataset_text(
#     dataset_name="takala/financial_phrasebank",
#     split="train",
#     text_field="sentence",
#     file_name="financial_phrasebank",
#     config_name="sentences_allagree",
#     trust_remote_code=True
# )
#
#
# # 3. Twitter Financial News - 10,000
# save_dataset_text(
#     "zeroshot/twitter-financial-news-sentiment",
#     "train",
#     "text",
#     "twitter_financial_news",
#     trust_remote_code=True,
#     limit=10000
# )
#
# # 4. AG News - 25,000
# save_dataset_text(
#     "ag_news",
#     "train",
#     "text",
#     "ag_news_en",
#     limit=25000
# )
#
# # Load and shuffle dataset
# ds_support = load_dataset("MohammadOthman/mo-customer-support-tweets-945k", split="train").shuffle(seed=42)
#
# # Select 50,000 entries
# ds_support = ds_support.select(range(50000))
#
# # Properly extract input and output fields
# inputs = [ex["input"].strip() for ex in ds_support if isinstance(ex, dict) and ex.get("input")]
# outputs = [ex["output"].strip() for ex in ds_support if isinstance(ex, dict) and ex.get("output")]
#
# # Take only first 25k of each
# inputs = inputs[:25000]
# outputs = outputs[:25000]
#
# # Save them
# with open("data/en/customer_support_inputs.txt", "w", encoding="utf-8") as f:
#     for line in inputs:
#         f.write(line + "\n")
#
# with open("data/en/customer_support_outputs.txt", "w", encoding="utf-8") as f:
#     for line in outputs:
#         f.write(line + "\n")
#
# print("✅ Saved customer support inputs and outputs.")

# 6. Wikipedia Paragraphs (~21k English entries)
save_dataset_text(
    dataset_name="agentlans/wikipedia-paragraphs",
    split="train",
    text_field="text",
    file_name="wikipedia_paragraphs"
)
