import re

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel


SENTENCE_MODEL = SentenceTransformer("bert-base-nli-mean-tokens")
WORD_TOKENIZER = BertTokenizer.from_pretrained("setu4993/LaBSE")
WORD_MODEL = BertModel.from_pretrained("setu4993/LaBSE")
TOKEN_PATTERN = re.compile(r"[A-Za-z]+")
EMBEDDING_CACHE = {}


def sentence_bert_embedding(sentence):
    """Return a sentence embedding for sentence-level similarity."""
    return SENTENCE_MODEL.encode(sentence).reshape(1, -1)


def cosine_similarity_sentences(vec1, vec2):
    """Return cosine similarity for two sentence embeddings."""
    return cosine_similarity(vec1, vec2)[0][0]


def normalized_word_set(sentence):
    """Return lowercase alphabetic words as a set."""
    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(sentence)}


def word_embeddings(sentence):
    """Return token strings and token embeddings for a sentence."""
    inputs = WORD_TOKENIZER(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = WORD_MODEL(**inputs)

    embeddings = outputs.last_hidden_state.squeeze(0)[1:-1].cpu().numpy()
    words = WORD_TOKENIZER.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))[1:-1]
    return words, embeddings


def cached_word_embeddings(sentence):
    """Cache token embeddings for a sentence."""
    if sentence not in EMBEDDING_CACHE:
        EMBEDDING_CACHE[sentence] = word_embeddings(sentence)
    return EMBEDDING_CACHE[sentence]


def semantic_token_overlap(sentence1, sentence2):
    """Return average best-match token similarity across two sentences."""
    words1, embeddings1 = cached_word_embeddings(sentence1)
    words2, embeddings2 = cached_word_embeddings(sentence2)

    if len(words1) == 0 or len(words2) == 0:
        return 0.0

    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    best_matches = np.max(similarity_matrix, axis=1)
    return float(np.sum(best_matches) / (len(words1) + len(words2)))


def jaccard_similarity(sentence1, sentence2):
    """Return Jaccard similarity for the word sets of two sentences."""
    words1 = normalized_word_set(sentence1)
    words2 = normalized_word_set(sentence2)

    union = words1 | words2
    if not union:
        return 0.0

    intersection = words1 & words2
    return len(intersection) / len(union)


def compare_sentences(sentence_a, sentence_b):
    """Return cosine, Jaccard, and semantic-overlap scores for two sentences."""
    emb_a = sentence_bert_embedding(sentence_a)
    emb_b = sentence_bert_embedding(sentence_b)

    return {
        "cosine_similarity_sentences_BERT": cosine_similarity_sentences(emb_a, emb_b),
        "jaccard_similarity": jaccard_similarity(sentence_a, sentence_b),
        "semantic_token_overlap": semantic_token_overlap(sentence_a, sentence_b),
    }
