from pathlib import Path
import csv

from transformers import BertTokenizer, BertModel
import torch
from sklearn.preprocessing import normalize


VOCAB_CSV_DIR = Path("vocabularies/csv_vocab")
BERT_EMBEDDING_SOURCE = "bert-base-uncased"

def embed_word(word, model, tokenizer, sentence=None):
    """
    Wrapper func for each mode
    """
    
    if not sentence:
        return embed_word_without_context(word, model, tokenizer)

    else:
        return embed_word_with_context(word, model, tokenizer, sentence)

    raise ValueError("Invalid embedding mode")

def embed_word_without_context(word, model, tokenizer):
    """
    Return a normalized embedding for a single word.
    """

    input_ids = torch.tensor(tokenizer.encode(word)).unsqueeze(0)
    outputs = model(input_ids)
    embedding = outputs[1].detach().numpy()
    return normalize(embedding)

def embed_word_with_context(sentence, target_word, model, tokenizer):
    """
    Return embedding for a single word given sentence's context
    """

    inputs = tokenizer(sentence, return_tensors="pt")
    outputs = model(**inputs)

    tokens = tokenizer.tokenize(sentence)
    token_embeddings = outputs.last_hidden_state[0]

    target_tokens = tokenizer.tokenize(target_word)

    for i in range(len(tokens)):
        if tokens[i:i+len(target_tokens)] == target_tokens:
            start = i
            end = i + len(target_tokens)
            break
    else:
        raise ValueError("Word not found")

    subword_vectors = token_embeddings[start+1:end+1]

    word_embedding = torch.mean(subword_vectors, dim=0)

    return word_embedding.detach().numpy()

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

    tokenizer = BertTokenizer.from_pretrained(BERT_EMBEDDING_SOURCE)
    model = BertModel.from_pretrained(BERT_EMBEDDING_SOURCE)
    return model, tokenizer


def build_vocab_embeddings(vocabulary, model, tokenizer):
    """
    Build Bert embeddings for the allowed vocabulary words.
    """

    vocab_embeddings = {}
    for word in vocabulary:
        vocab_embeddings[word] = embed_word_without_context(word, model, tokenizer).reshape(-1)
    return vocab_embeddings
