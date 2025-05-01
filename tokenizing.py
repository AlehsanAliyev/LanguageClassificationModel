from tokenizers import Tokenizer, trainers, models, pre_tokenizers, processors
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from tokenizers.pre_tokenizers import Whitespace
import pandas as pd
import os

# Load train text
train_df = pd.read_csv("data/final/train.csv")

# Save only text to train BPE on
with open("data/final/train_text.txt", "w", encoding="utf-8") as f:
    for line in train_df["text"]:
        f.write(line.strip() + "\n")

# === Train BPE Tokenizer ===
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(vocab_size=8000, special_tokens=["[PAD]", "[UNK]"])

tokenizer.train(["data/final/train_text.txt"], trainer)
tokenizer.save("data/final/bpe_tokenizer.json")

print("✅ Tokenizer trained and saved.")

# === Encode train/test data ===
def encode_texts(df, tokenizer, max_length=128):
    encoded = [tokenizer.encode(text.strip()).ids[:max_length] for text in df["text"]]
    return pd.DataFrame({
        "input_ids": [" ".join(map(str, ids)) for ids in encoded],
        "label": df["label"]
    })

tokenizer = Tokenizer.from_file("data/final/bpe_tokenizer.json")
test_df = pd.read_csv("data/final/test.csv")

df_train_enc = encode_texts(train_df, tokenizer)
df_test_enc = encode_texts(test_df, tokenizer)

df_train_enc.to_csv("data/final/train_tokenized.csv", index=False)
df_test_enc.to_csv("data/final/test_tokenized.csv", index=False)

print("✅ Tokenized train/test data saved.")
