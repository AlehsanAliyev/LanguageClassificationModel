import ollama

# LLaMA 3 model name — must match what you've pulled in Ollama (e.g., llama3, llama3:8b)
MODEL_NAME = "llama3"

# Your transliterated Russian texts
texts = [
    "privet", "poka", "spasibo", "pojaluysta", "khorosho", "plokho", "dom", "rabota", "voda", "drujba",
    "kak dela", "ya lyublyu tebya", "eto ochen vkusno", "ya ne znayu", "kak tebya zovut", "dobroye utro",
    "dobriy vecher", "mne nujna pomoshch", "gde tualet", "skolko eto stoit",
    "privet, menya zovut Ivan", "segodnya khoroshaya pogoda", "ya zhivu v Moskve", "ona rabotaet v bolnitse",
    "my idyom v kino segodnya vecherom", "mne nravitsya russkaya eda", "u menya net deneg", "ya izuchayu angliyskiy yazyk",
    "vchera my bili na vecherinke", "on chitayet knigu v parke"
]

def ask_ollama(text):
    prompt = f"""Detect the language of the following sentence. Only reply with a language code: "az" for Azerbaijani, "en" for English, or "ru" for Russian.

Text: "{text}"
Language code:"""

    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return response['message']['content'].strip().lower()

# Run inference
results = {}
for text in texts:
    lang = ask_ollama(text)
    print(f"{text} -> {lang}")
    results[lang] = results.get(lang, 0) + 1

# Summary
print("\n📊 Results:")
for lang, count in results.items():
    print(f"{lang}: {count}")
