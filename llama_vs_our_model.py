import torch
import pandas as pd
from tokenizers import Tokenizer
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# === Setup ===
MODEL_PATH = "models/bilstm_langid3.pt"
TOKENIZER_PATH = "data/final2/bpe_tokenizer.json"
TEST_CSV = "data/final2/test.csv"

label2id = {"az": 0, "en": 1, "ru": 2}
id2label = {v: k for k, v in label2id.items()}
MAX_LEN = 100

# === Load tokenizer ===
tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

# === Load test data ===
df = pd.read_csv(TEST_CSV)
texts = df["text"].tolist()
true_labels = df["label"].tolist()


# === BiLSTM Model ===
class BiLSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        h_cat = torch.cat((h_n[0], h_n[1]), dim=1)
        return self.fc(h_cat)


# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMClassifier(64000, 1024, 512, 3).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()


# === Predict using BiLSTM ===
def predict_bilstm(text):
    ids = tokenizer.encode(text.strip()).ids[:MAX_LEN]
    padded = ids + [0] * (MAX_LEN - len(ids))
    x = torch.tensor([padded], dtype=torch.long).to(device)
    with torch.no_grad():
        pred = model(x).argmax(dim=1).item()
    return pred


# === Run predictions with progress bar ===
bilstm_preds = [predict_bilstm(text) for text in tqdm(texts, desc="Predicting with BiLSTM")]


# === Evaluate and save confusion matrix ===
def evaluate(preds, name):
    print(f"\n📊 {name} Accuracy Report")
    print(classification_report(true_labels, preds, target_names=["az", "en", "ru"]))

    cm = confusion_matrix(true_labels, preds)
    print("Confusion Matrix:\n", cm)

    # Plot and save confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=["az", "en", "ru"], yticklabels=["az", "en", "ru"], cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig("logs/confusion_matrix.png")
    print("✅ Confusion matrix saved to logs/confusion_matrix.png")


evaluate(bilstm_preds, "BiLSTM")

# === Save mismatches ===
mismatches = []
for text, true, pred in zip(texts, true_labels, bilstm_preds):
    if true != pred:
        mismatches.append([text, id2label[true], id2label[pred]])

pd.DataFrame(mismatches, columns=["text", "true", "bilstm"]).to_csv("logs/mismatches.csv", index=False)
print("⚠️ Mismatches saved to logs/mismatches.csv")
