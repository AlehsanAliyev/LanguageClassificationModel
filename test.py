# import torch
# print(torch.cuda.is_available())

import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import pandas as pd


# ========== Check for GPU ==========
if not torch.cuda.is_available():
    print("❌ CUDA GPU not available. Training stopped.")
    sys.exit(1)

device = torch.device("cuda")
print(f"✅ Using device: {device}")

# ========== Parameters ==========
VOCAB_SIZE = 64000  # updated BPE vocab size
EMBED_DIM = 1024
HIDDEN_DIM = 512
NUM_CLASSES = 3
BATCH_SIZE = 64
EPOCHS = 5

# ========== Load Tokenized CSVs ==========
def load_dataset(path):
    df = pd.read_csv(path)
    sequences = df["input_ids"].apply(lambda x: [int(i) for i in x.strip().split()])
    labels = df["label"].tolist()
    max_len = max(len(seq) for seq in sequences)
    print("Max_length:", max_len)
    padded = [seq + [0] * (max_len - len(seq)) for seq in sequences]
    x_tensor = torch.tensor(padded, dtype=torch.long)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    return x_tensor, y_tensor

x_train, y_train = load_dataset("data/final2/train_tokenized2.csv")
x_test, y_test = load_dataset("data/final2/test_tokenized2.csv")
