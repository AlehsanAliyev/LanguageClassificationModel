import os
from datasets import load_dataset

# Create folders
os.makedirs("data/az", exist_ok=True)

# Helper function to write to file
def save_texts(file_path, texts):
    with open(file_path, "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line.strip() + "\n")

# 1. Load and save azbanks-qadata
azbanks = load_dataset("arzumanabbasov/azbanks-qadata", split="train")
azbanks_texts = [item["question"] for item in azbanks if "question" in item and item["question"]]
save_texts("data/az/azbanks.txt", azbanks_texts)

# 2. Load and save anl-news subset from DOLLMA
anl_news = load_dataset("allmalab/DOLLMA", "anl-news", split="train")
anl_news_texts = [item["text"] for item in anl_news if "text" in item and item["text"]]
save_texts("data/az/anl-news.txt", anl_news_texts)

# 3. Load and save azwiki subset from DOLLMA
azwiki = load_dataset("allmalab/DOLLMA", "azwiki", split="train")
azwiki_texts = [item["text"] for item in azwiki if "text" in item and item["text"]]
save_texts("data/az/azwiki.txt", azwiki_texts)

print("✅ Azerbaijani datasets saved in 'data/az/'")
