# Load dependencies
import torch
from tokenizers import Tokenizer

# Load tokenizer and model
tokenizer = Tokenizer.from_file("data/final2/bpe_tokenizer.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model (same as in training)
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

# Load model
model = BiLSTMClassifier(64000, 1024, 512, 3).to(device)
model.load_state_dict(torch.load("models/bilstm_langid3.pt", map_location=device))
model.eval()

# Label mapping
id2label = {0: "az", 1: "en", 2: "ru"}

single_words = [
    "privet",            # hi
    "poka",              # bye
    "spasibo",           # thank you
    "pojaluysta",       # please
    "khorosho",          # good
    "plokho",             # bad
    "dom",               # house
    "rabota",            # work
    "voda",              # water
    "drujba"            # friendship
]
phrases = [
    "kak dela",                        # how are you
    "ya lyublyu tebya",                # I love you
    "eto ochen vkusno",               # this is very tasty
    "ya ne znayu",                     # I don't know
    "kak tebya zovut",                # what is your name
    "dobroye utro",                    # good morning
    "dobriy vecher",                  # good evening
    "mne nujna pomoshch",            # I need help
    "gde tualet",                     # where is the toilet
    "skolko eto stoit"                # how much does it cost
]
sentences = [
    "privet, menya zovut Ivan",               # Hello, my name is Ivan
    "segodnya khoroshaya pogoda",             # Today the weather is good
    "ya zhivu v Moskve",                      # I live in Moscow
    "ona rabotaet v bolnitse",                # She works in a hospital
    "my idyom v kino segodnya vecherom",      # We are going to the cinema tonight
    "mne nravitsya russkaya eda",             # I like Russian food
    "u menya net deneg",                      # I have no money
    "ya izuchayu angliyskiy yazyk",           # I am learning English
    "vchera my bili na vecherinke",             # Yesterday we were at a party
    "on chitayet knigu v parke"               # He is reading a book in the park
]

all_texts = single_words + phrases + sentences


ans = {}
# Inference
MAX_LEN = 100
for text in all_texts:
    ids = tokenizer.encode(text).ids[:MAX_LEN]
    padded = ids + [0] * (MAX_LEN - len(ids))
    x = torch.tensor([padded], dtype=torch.long).to(device)
    with torch.no_grad():
        pred = model(x).argmax(dim=1).item()
    print(f"{text} -> {id2label[pred]}")
    if id2label[pred] not in ans:
        ans[id2label[pred]] = 1
    else:
        ans[id2label[pred]] += 1


print("Overall accuracy: ", ans["ru"]/len(all_texts))
print("Detected languages: ", ans)

