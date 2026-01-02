import os
import sys
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
import pandas as pd

# ========== Logging ==========
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"train_bilstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

class Logger:
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w", encoding="utf-8")
    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)

# ========== Check for GPU ==========
if not torch.cuda.is_available():
    print("CUDA GPU not available. Training stopped.")
    sys.exit(1)

device = torch.device("cuda")
print(f"Using device: {device}")

# ========== Parameters ==========
VOCAB_SIZE = 64000
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
    max_len = 1024
    padded = [seq + [0] * (max_len - len(seq)) for seq in sequences]
    x_tensor = torch.tensor(padded, dtype=torch.long)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    return x_tensor, y_tensor

x_train, y_train = load_dataset("data/final2/train_tokenized2.csv")
x_test, y_test = load_dataset("data/final2/test_tokenized2.csv")

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=BATCH_SIZE)

# ========== BiLSTM Model ==========
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        h_cat = torch.cat((h_n[0], h_n[1]), dim=1)
        return self.fc(h_cat)

model = BiLSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# ========== Training ==========
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

# ========== Save Model ==========
os.makedirs("models", exist_ok=True)
model_path = "models/bilstm_langid4.pt"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# ========== Reload & Evaluate ==========
model.load_state_dict(torch.load(model_path))
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        preds = model(x_batch).argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += y_batch.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
print(f"Training log saved to: {log_file}")
