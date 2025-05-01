import gradio as gr
import torch
import torch.nn as nn
import pandas as pd
from tokenizers import Tokenizer

# ==== Load Tokenizer ====
tokenizer = Tokenizer.from_file("data/final/bpe_tokenizer.json")
MAX_LEN = 100

# ==== Label Mapping ====
id2label = {0: "az", 1: "en", 2: "ru"}

# ==== Model Config ====
VOCAB_SIZE = 30000  # Should match training
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_CLASSES = 3

# ==== Model Definition ====
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        h_cat = torch.cat((h_n[0], h_n[1]), dim=1)
        return self.fc(h_cat)

# ==== Load Model ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMClassifier(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES).to(device)
model.load_state_dict(torch.load("models/bilstm_langid.pt", map_location=device))
model.eval()

# ==== Prediction Function ====
def predict_language(text):
    encoding = tokenizer.encode(text)
    input_ids = encoding.ids[:MAX_LEN]
    padded = input_ids + [0] * (MAX_LEN - len(input_ids))
    tensor = torch.tensor([padded], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(dim=1).item()
    return id2label[pred]

# ==== Launch Gradio Interface ====
demo = gr.Interface(
    fn=predict_language,
    inputs=gr.Textbox(label="Enter Text"),
    outputs=gr.Label(label="Predicted Language"),
    title="🌐 Language Identifier",
    description="Enter a sentence and predict whether it's Azerbaijani (az), English (en), or Russian (ru).",
    examples=[
        "Bu gün hava çox gözəldir.",
        "This model was trained on multilingual data.",
        "Привет, как дела?"
    ]
)

demo.launch()
