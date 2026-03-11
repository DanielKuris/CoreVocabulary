from pathlib import Path
import csv

from transformers import BertTokenizer, BertModel
import torch
from sklearn.preprocessing import normalize


VOCAB_CSV_DIR = Path("vocabularies/csv_vocab")
LABSE_EMBEDDING_SOURCE = "setu4993/LaBSE"


def embed_word(word, model, tokenizer):
    """
    Return a normalized embedding for a single word.
    """

    input_ids = torch.tensor(tokenizer.encode(word)).unsqueeze(0)
    outputs = model(input_ids)
    embedding = outputs[1].detach().numpy()
    return normalize(embedding)


def vocabulary_csv_path(vocabulary_size):
    """
    Return the CSV path for a configured vocabulary size.
    """

    return VOCAB_CSV_DIR / f"vocab{vocabulary_size}.csv"


def load_vocabulary(vocabulary_size):
    """
    Load the allowed vocabulary list from the configured CSV file.
    """

    vocabulary_path = vocabulary_csv_path(vocabulary_size)
    if not vocabulary_path.exists():
        raise FileNotFoundError(f"Vocabulary CSV not found: {vocabulary_path}")

    vocabulary = []
    with open(vocabulary_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 2:
                vocabulary.append(row[1].strip().lower())

    return vocabulary


def load_replacement_model():
    """
    Load the word-level model used for nearest-vocabulary replacement.
    """

    tokenizer = BertTokenizer.from_pretrained(LABSE_EMBEDDING_SOURCE)
    model = BertModel.from_pretrained(LABSE_EMBEDDING_SOURCE)
    return model, tokenizer


def build_vocab_embeddings(vocabulary, model, tokenizer):
    """
    Build LaBSE embeddings for the allowed vocabulary words.
    """

    vocab_embeddings = {}
    for word in vocabulary:
        vocab_embeddings[word] = embed_word(word, model, tokenizer).reshape(-1)
    return vocab_embeddings
