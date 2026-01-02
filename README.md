# Language Classification Model
A BiLSTM-based language identifier for Azerbaijani (az), English (en), and Russian (ru).

## What it does
- Trains a BiLSTM classifier on word-, sentence-, and multi-sentence corpora.
- Builds and applies a BPE tokenizer for text normalization and encoding.
- Serves a local Gradio demo for interactive language prediction.
- Evaluates predictions with classification reports and confusion matrices.

## Datasets
Used by default in the current data download scripts:
- Azerbaijani: `arzumanabbasov/azbanks-qadata`, `allmalab/DOLLMA` (configs: `anl-news`, `azwiki`) in `load_aze_data.py`.
- English: `agentlans/wikipedia-paragraphs` in `load_english_data.py`.
- Russian: `FacelessLake/noise-augmented-russian-librispeech`, `Romjiik/Russian_bank_reviews`, `Den4ikAI/russian_cleared_wikipedia` in `load_rus_data.py`.

Optional datasets present in `load_english_data.py` (commented out by default):
- `wikimedia/wikipedia` (config: `20231101.ace`)
- `takala/financial_phrasebank` (config: `sentences_allagree`)
- `zeroshot/twitter-financial-news-sentiment`
- `ag_news`
- `MohammadOthman/mo-customer-support-tweets-945k`

## Demo (Gradio) quickstart
Requires a trained model at `models/bilstm_langid2.pt` and a tokenizer at `data/final2/bpe_tokenizer.json` (as loaded in `gradio_app.py`).

```powershell
# optional: create/activate a virtual environment
python -m venv venv
.\venv\Scripts\activate

# install dependencies
pip install torch pandas tokenizers datasets scikit-learn matplotlib seaborn tqdm gradio ollama

# run demo
python gradio_app.py
```

## Training
The training pipeline in this repo is driven by the existing scripts below. Dataset downloads require internet access and use Hugging Face datasets.

```powershell
# 1) Download raw data
python load_aze_data.py
python load_english_data.py
python load_rus_data.py

# 2) Optional: transliterate Russian corpora (used by sentence_corpus.py)
python transliterate_rus.py

# 3) Build top words and word-level corpus
python top500_in_aze.py
python top500_en.py
python top500_ru.py
python word_level.py

# 4) Build sentence and multi-sentence corpora
python sentence_corpus.py

# 5) Prepare train/test split
python prepare_data_for_classification.py

# 6) Train tokenizer and tokenize datasets
python tokenizing.py

# 7) Train BiLSTM model (CUDA required)
python BiLSTM_Classifier.py
```

Evaluation (optional):
```powershell
python llama_vs_our_model.py
```

## Inference / prediction usage
This snippet mirrors the inference logic in `gradio_app.py`.

```python
import torch
import torch.nn as nn
from tokenizers import Tokenizer

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

tokenizer = Tokenizer.from_file("data/final2/bpe_tokenizer.json")
id2label = {0: "az", 1: "en", 2: "ru"}
MAX_LEN = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMClassifier(64000, 128, 128, 3).to(device)
model.load_state_dict(torch.load("models/bilstm_langid2.pt", map_location=device))
model.eval()

def predict_language(text: str) -> str:
    ids = tokenizer.encode(text).ids[:MAX_LEN]
    padded = ids + [0] * (MAX_LEN - len(ids))
    x = torch.tensor([padded], dtype=torch.long).to(device)
    with torch.no_grad():
        pred = model(x).argmax(dim=1).item()
    return id2label[pred]

print(predict_language("salam, bu gun hava yaxsidir"))
print(predict_language("this is a simple test sentence"))
print(predict_language("privet, kak dela"))
```

## Data
Expected locations and formats based on the scripts:

- Raw language text files:
  - `data/az/*.txt`, `data/en/*.txt`, `data/ru/*.txt`
- Top word lists:
  - `data/az/top500_words/*.txt`, `data/en/top500_words/*.txt`, `data/ru/top_words2/*.txt`
- Word-level corpus CSV:
  - `data/corpus/word_corpus2.csv` with columns: `text, lang`
- Sentence corpora CSVs:
  - `data/corpus/sentence_corpus_balanced2.csv`
  - `data/corpus/multisentence_corpus_balanced2.csv`
  - Both with columns: `text, lang`
- Train/test CSVs:
  - `data/final2/train.csv`, `data/final2/test.csv` with columns: `text, label`
- Tokenized CSVs:
  - `data/final2/train_tokenized2.csv`, `data/final2/test_tokenized2.csv` with columns: `input_ids, label`

Label space is defined in multiple scripts as:
- `az` -> 0
- `en` -> 1
- `ru` -> 2

## Project structure

| Path | Purpose |
| --- | --- |
| `.gitignore` | Ignores large artifacts and data outputs |
| `BiLSTM_Classifier.py` | BiLSTM training and evaluation |
| `gradio_app.py` | Gradio demo for local inference |
| `tokenizing.py` | Train BPE tokenizer and tokenize datasets |
| `prepare_data_for_classification.py` | Build train/test CSVs |
| `sentence_corpus.py` | Build sentence/multi-sentence corpora |
| `word_level.py` | Build word-level corpus |
| `load_aze_data.py` | Download Azerbaijani datasets |
| `load_english_data.py` | Download English datasets |
| `load_rus_data.py` | Download Russian datasets |
| `transliterate_rus.py` | Transliterates Russian text |
| `custom_russian_transliteraion.py` | Custom transliteration rules |
| `llama_vs_our_model.py` | Evaluate predictions + confusion matrix |
| `test_russian_transliteraion.py` | Manual inference on transliterated Russian |
| `test.py` | Tokenized dataset inspection (CUDA required) |
| `models/` | Saved PyTorch weights (ignored in git) |
| `data/` | Raw and processed data (ignored in git) |
| `logs/` | Training/eval logs (ignored in git) |



## Roadmap / TODO
- Align training and inference hyperparameters via a shared config.
- Add CLI arguments for data/model paths.
- Add CPU fallback for training instead of exiting when CUDA is unavailable.


## Acknowledgements
- Hugging Face datasets used in `load_aze_data.py`, `load_english_data.py`, and `load_rus_data.py`:
  - https://huggingface.co/datasets/arzumanabbasov/azbanks-qadata
  - https://huggingface.co/datasets/allmalab/DOLLMA (configs: `anl-news`, `azwiki`)
  - https://huggingface.co/datasets/agentlans/wikipedia-paragraphs
  - https://huggingface.co/datasets/FacelessLake/noise-augmented-russian-librispeech
  - https://huggingface.co/datasets/Romjiik/Russian_bank_reviews
  - https://huggingface.co/datasets/Den4ikAI/russian_cleared_wikipedia
  - Optional (commented in `load_english_data.py`):
    - https://huggingface.co/datasets/wikimedia/wikipedia (config: `20231101.ace`)
    - https://huggingface.co/datasets/takala/financial_phrasebank (config: `sentences_allagree`)
    - https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment
    - https://huggingface.co/datasets/ag_news
    - https://huggingface.co/datasets/MohammadOthman/mo-customer-support-tweets-945k
- Open-source libraries: PyTorch, Hugging Face tokenizers/datasets, Gradio, scikit-learn, matplotlib.

