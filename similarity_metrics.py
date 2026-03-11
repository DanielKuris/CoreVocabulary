from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch
import numpy as np


SENTENCE_MODEL = SentenceTransformer("bert-base-nli-mean-tokens")
WORD_TOKENIZER = BertTokenizer.from_pretrained("setu4993/LaBSE")
WORD_MODEL = BertModel.from_pretrained("setu4993/LaBSE")
EMBEDDING_CACHE = {}


def sentence_bert_embedding(sentence):
    """Return a sentence embedding for sentence-level similarity."""
    return SENTENCE_MODEL.encode(sentence).reshape(1, -1)


def cosine_similarity_sentences(vec1, vec2):
    """Return cosine similarity for two sentence embeddings."""
    return cosine_similarity(vec1, vec2)[0][0]


def word_bert_embeddings(sentence):
    """Return token strings and token embeddings for a sentence."""
    inputs = WORD_TOKENIZER(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = WORD_MODEL(**inputs)

    embeddings = outputs.last_hidden_state.squeeze(0)
    embeddings = embeddings[1:-1].cpu().numpy()
    words = WORD_TOKENIZER.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))[1:-1]
    return words, embeddings


def cached_word_embeddings(sentence):
    """Cache token embeddings for a sentence."""
    if sentence not in EMBEDDING_CACHE:
        EMBEDDING_CACHE[sentence] = word_bert_embeddings(sentence)
    return EMBEDDING_CACHE[sentence]


def jaccard_similarity_bert(sentence1, sentence2, threshold=0.7):
    """Return a token-overlap score based on embedding similarity.

    For each token in sentence1, the function finds the best token-level match in
    sentence2. Only matches at or above ``threshold`` contribute to the score.
    """
    words1, embeddings1 = cached_word_embeddings(sentence1)
    words2, embeddings2 = cached_word_embeddings(sentence2)

    if len(words1) == 0 or len(words2) == 0:
        return 0.0

    similarity_matrix = cosine_similarity(embeddings1, embeddings2)

    intersection_similarity = 0
    for row in similarity_matrix:
        best_match = np.max(row)
        if best_match >= threshold:
            intersection_similarity += best_match

    union_size = len(words1) + len(words2)
    return intersection_similarity / union_size


def compare_sentences(sentence_a, sentence_b):
    """Return cosine and token-overlap similarity scores for two sentences."""
    emb_a = sentence_bert_embedding(sentence_a)
    emb_b = sentence_bert_embedding(sentence_b)

    return {
        "cosine_similarity_sentences_BERT": cosine_similarity_sentences(emb_a, emb_b),
        "jaccard_similarity_BERT": jaccard_similarity_bert(sentence_a, sentence_b),
    }
