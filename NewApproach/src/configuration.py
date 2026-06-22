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
    "T5": True,           # Model B: T5-Small (Negative Token Masking)
    "BYT5": True,         # Model C: ByT5-Small (Positive Character Trie)
    "EMB_SUB": True       # Model D: Embedding Vector Substitution Model
}

MODEL_NAMES = {
    "MLM": "distilbert-base-uncased",
    "T5": "t5-small",
    "BYT5": "google/byt5-small",
    "EMB_SUB": "distilbert-base-uncased"
}
# =====================================================================
# 🧮 2. METRICS SELECTION MODULE
# =====================================================================
METRIC_SELECTION = {
    "SARI": True,
    "BLEU": True,
    "METEOR": True,
    "BERTScore": True,
    "JACCARD": True,      # Jaccard token-level intersection over union matrix
    "COSINE": True        # DistilBERT contextual vector cosine mapping
}

# =====================================================================
# 📚 3. EXEMPT VOCABULARY CONFIGURATION
# Options: "english_stopwords" (NLTK set) or "none"
# =====================================================================
# Multi-run configurations: set each run strategy to True or False.
# If all strategies are set to False, the system automatically defaults to "none".
EXEMPT_RUNS = {
    "english_stopwords": True,
    "none": True
}

# Resolve active running states dynamically for downstream pipeline iterations
ACTIVE_EXEMPT_MODES = [mode for mode, enabled in EXEMPT_RUNS.items() if enabled]
if not ACTIVE_EXEMPT_MODES:
    ACTIVE_EXEMPT_MODES = ["none"]

# Base static punctuation mappings always excluded from lexical constraint sweeps
BASE_PUNCTUATION = {".", ",", "!", "?", ";", ":", "-", "_", "(", ")", '"', "'"}

# Helper function to extract vocabulary sets dynamically matching legacy logic
def get_exempt_vocabulary(mode: str) -> set:
    if mode == "english_stopwords":
        return set(stopwords.words('english')).union(BASE_PUNCTUATION)
    return BASE_PUNCTUATION

# =====================================================================
# 📊 4. TARGET VOCABULARIES CONFIGURATION (Updated paths)
# =====================================================================
# Hardcoded reference baselines
VOCAB_FILES = {
    "src/vocabularies/ThingExplainer1000.txt": True, 
    "src/vocabularies/OgdenBasicEnglish850.txt": True
}

# 🚀 DYNAMIC GENERATION: Automatically append vocab1_100.txt through vocab1_2000.txt
for size in range(100, 2100, 100):
    vocab_path = f"src/vocabularies/vocab1_{size}.txt"
    VOCAB_FILES[vocab_path] = True

# =====================================================================
# 🧱 5. BENCHMARK DATASET SELECTION
# =====================================================================
DATASETS = {
    "turk_corpus": True,
    "asset": True,
    "sick": True         # Sentence Involvement & Compositional Knowledge benchmark
}

# 🆕 Dynamic SICK control configurations for future iterations
SICK_SLICING = 400  # Set to an integer (e.g., 400) or None if you want to run the full 9,840 split