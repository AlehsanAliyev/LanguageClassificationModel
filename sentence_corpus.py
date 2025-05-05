import os
import csv
import re
import random
import sys
from datetime import datetime
from collections import defaultdict

# ========= Logging Setup =========
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"corpus_build_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

class Logger:
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)

# ========= Settings =========
random.seed(42)
LANG_LIMITS = {
    "az": {
        "max_sent": 50000, "max_multi": 25000,
        "limits": {
            "data/az/azbanks.txt": (20000, 5000),
            "data/az/azwiki.txt": (25000, 15000),
            "data/az/anl-news.txt": ("fill", "fill"),
        }
    },
    "en": {
        "max_sent": 50000, "max_multi": 25000,
        "limits": {
            "data/en/customer_support_inputs.txt": (11000, 8000),
            "data/en/customer_support_outputs.txt": (14000, 7000),
            "data/en/ag_news_en.txt": (15000, 5000),
            "data/en/wikipedia_paragraphs.txt": ("fill", "fill")
        }
    },
    "ru": {
        "max_sent": 50000, "max_multi": 25000,
        "limits": {
            "data/ru/bank_reviews.txt": (15000, 9000),
            "data/ru/librispeech_text.txt": (7000, 1000),
            "data/ru/russian_cleared_wikipedia.txt": (8000, 6000),
            "data/ru/bank_reviews_translit2.txt": (10000, 5000),
            "data/ru/russian_cleared_wikipedia_translit2.txt": ("fill", "fill")
        }
    }
}

# Corpus sources
corpus_sources = {
    "az": [
        ("data/az/azbanks.txt", True),
        ("data/az/azwiki.txt", True),
        ("data/az/anl-news.txt", True)
    ],
    "en": [
        ("data/en/customer_support_inputs.txt", False),
        ("data/en/customer_support_outputs.txt", False),
        ("data/en/ag_news_en.txt", False),
        ("data/en/wikipedia_paragraphs.txt", False)
    ],
    "ru": [
        ("data/ru/bank_reviews.txt", False),
        ("data/ru/librispeech_text.txt", False),
        ("data/ru/russian_cleared_wikipedia.txt", False),
        ("data/ru/bank_reviews_translit2.txt", False),
        ("data/ru/russian_cleared_wikipedia_translit2.txt", False)
    ]
}

# Sentence splitting
def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())

# Corpus storage
sent_rows = []
multi_rows = []
used_sents = defaultdict(set)

# Build corpus
for lang, sources in corpus_sources.items():
    sent_count = 0
    multi_count = 0
    print(f"\n\U0001F524 Starting language: {lang.upper()}")

    for path, is_az in sources:
        if not os.path.exists(path):
            print(f"⚠️  File not found: {path}")
            continue

        rel_sent, rel_multi = LANG_LIMITS[lang]["limits"][path]
        limit_sent = rel_sent if rel_sent != "fill" else LANG_LIMITS[lang]["max_sent"] - sent_count
        limit_multi = rel_multi if rel_multi != "fill" else LANG_LIMITS[lang]["max_multi"] - multi_count

        print(f"📄 Reading from: {path} (AZ mode: {is_az})")

        local_sent, local_multi = 0, 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if local_sent >= limit_sent and local_multi >= limit_multi:
                    print(f"⏹️  Reached limit for {path} — skipping rest")
                    break

                line = line.strip()
                if not line:
                    continue

                sents = split_sentences(line)
                sents = [s.strip() for s in sents if len(s.split()) >= 2]
                if not sents:
                    continue

                if is_az:
                    if random.random() < 0.5 and len(sents) >= 2 and local_multi < limit_multi:
                        idx = random.randint(0, len(sents) - 2)
                        chunk = " ".join(sents[idx:idx+3])
                        multi_rows.append((chunk, lang))
                        local_multi += 1
                        multi_count += 1
                    elif local_sent < limit_sent:
                        s = random.choice(sents)
                        sent_rows.append((s, lang))
                        local_sent += 1
                        sent_count += 1
                else:
                    for i in range(len(sents) - 1):
                        if local_multi >= limit_multi:
                            break
                        chunk = " ".join(sents[i:i+3])
                        multi_rows.append((chunk, lang))
                        used_sents[lang].update(sents[i:i+3])
                        local_multi += 1
                        multi_count += 1

                    for s in sents:
                        if local_sent >= limit_sent:
                            break
                        if s not in used_sents[lang]:
                            sent_rows.append((s, lang))
                            local_sent += 1
                            sent_count += 1

        print(f"✅ Finished {path} — Added: {local_sent} sentences, {local_multi} multi-sentences")

    print(f"🏁 Completed {lang.upper()} — Final: {sent_count} sentences, {multi_count} multi-sentences")

# Save outputs
os.makedirs("data/corpus", exist_ok=True)

with open("data/corpus/sentence_corpus_balanced2.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows([["text", "lang"]] + sent_rows)

with open("data/corpus/multisentence_corpus_balanced2.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows([["text", "lang"]] + multi_rows)

print("\n✅ Corpus build complete!")
print(f"📝 Sentence corpus: {len(sent_rows)} rows")
print(f"📘 Multi-sentence corpus: {len(multi_rows)} rows")
print(f"🗂️  Log saved to: {log_file}")
