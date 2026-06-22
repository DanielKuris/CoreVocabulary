import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords

# =====================================================================
# 🧠 1. MODEL SELECTION MODULE
# =====================================================================
MODEL_SELECTION = {
    "MLM": True,          # Model A: DistilBERT (Template Substitution)
    "T5": True,          # Model B: T5-Small (Negative Token Masking)
    "BYT5": True          # Model C: ByT5-Small (Positive Character Trie)
}

# =====================================================================
# 🧮 2. METRICS SELECTION MODULE
# =====================================================================
METRIC_SELECTION = {
    "SARI": True,
    "BLEU": True,
    "METEOR": True,
    "BERTScore": True
}

# =====================================================================
# 📚 3. EXEMPT VOCABULARY CONFIGURATION
# Options: "english_stopwords" (NLTK set) or "none"
# =====================================================================
EXEMPT_MODE = "None"  # Change to "english_stopwords" to exempt common stopwords from constraints

if EXEMPT_MODE == "english_stopwords":
    EXEMPT_VOCABULARY = set(stopwords.words('english'))
else:
    EXEMPT_VOCABULARY = set()

# =====================================================================
# 📊 4. TARGET VOCABULARIES CONFIGURATION (Updated paths)
# =====================================================================
VOCAB_FILES = {
    "src/vocabularies/vocabulary.txt": True, 
    "src/vocabularies/ThingExplainer1000.txt": True, 
    "src/vocabularies/OgdenBasicEnglish850.txt": True
}

# =====================================================================
# 🧱 5. BENCHMARK DATASET SELECTION
# =====================================================================
DATASETS = {
    "turk_corpus": True,
    "asset": True
}