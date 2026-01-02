import os
import pandas as pd
from tokenizers import Tokenizer, trainers, models, pre_tokenizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence

# Paths
data_dir = "data/final2"
train_csv = os.path.join(data_dir, "train.csv")
test_csv = os.path.join(data_dir, "test.csv")
train_text_path = os.path.join(data_dir, "train_text2.txt")
tokenizer_path = os.path.join(data_dir, "bpe_tokenizer2.json")
tokenized_train_path = os.path.join(data_dir, "train_tokenized2.csv")
tokenized_test_path = os.path.join(data_dir, "test_tokenized2.csv")

# === Load and Save Train Text ===
train_df = pd.read_csv(train_csv)
with open(train_text_path, "w", encoding="utf-8") as f:
    for line in train_df["text"]:
        f.write(line.strip() + "\n")

# === Train BPE Tokenizer ===
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=64000,
    special_tokens=["[PAD]", "[UNK]"]
)

tokenizer.train([train_text_path], trainer)
tokenizer.save(tokenizer_path)

print("Tokenizer trained and saved.")

# === Encode Function ===
def encode_texts(df, tokenizer, max_length=1024):
    encoded = [tokenizer.encode(text.strip()).ids[:max_length] for text in df["text"]]
    return pd.DataFrame({
        "input_ids": [" ".join(map(str, ids)) for ids in encoded],
        "label": df["label"]
    })

# === Encode and Save ===
tokenizer = Tokenizer.from_file(tokenizer_path)
test_df = pd.read_csv(test_csv)

df_train_enc = encode_texts(train_df, tokenizer)
df_test_enc = encode_texts(test_df, tokenizer)

df_train_enc.to_csv(tokenized_train_path, index=False)
df_test_enc.to_csv(tokenized_test_path, index=False)

print("Tokenized train/test data saved.")
