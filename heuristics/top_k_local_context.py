import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from heuristics.local_context import context_vector


def top_vocabulary_words(target_vector, vocab_embeddings, top_k):
    """
    Return the top-k nearest vocabulary words for a target vector.
    """

    vocabulary_words = list(vocab_embeddings.keys())
    vocabulary_matrix = np.vstack(list(vocab_embeddings.values()))
    similarities = cosine_similarity(target_vector.reshape(1, -1), vocabulary_matrix)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]

    return [
        (vocabulary_words[index], similarities[index])
        for index in top_indices
    ]


def best_reranked_word(word_vector, current_context_vector, vocab_embeddings, top_k, context_weight):
    """
    Return the best candidate after nearest-neighbor shortlist reranking.
    """

    shortlist = top_vocabulary_words(word_vector, vocab_embeddings, top_k)
    best_word = shortlist[0][0]
    best_score = float("-inf")

    for candidate_word, base_similarity in shortlist:
        candidate_vector = np.asarray(vocab_embeddings[candidate_word]).reshape(1, -1)
        context_similarity = cosine_similarity(
            current_context_vector.reshape(1, -1),
            candidate_vector,
        )[0][0]
        score = ((1 - context_weight) * base_similarity) + (context_weight * context_similarity)
        if score > best_score:
            best_score = score
            best_word = candidate_word

    return best_word


def build_replacer(words, runtime, config):
    """
    Build a replacer that reranks top-k nearest candidates with context.
    """

    vocab_embeddings = runtime["vocab_embeddings"]
    embed_word = runtime["embed_word"]
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    context_weight = float(config.get("local_context_weight", 0.0))
    window_size = int(config.get("local_context_window", 3))
    top_k = int(config.get("top_k_candidates", 5))

    def replace_word(word, index):
        """
        Replace a word using top-k reranking with context.
        """

        word_vector = embed_word(word, model, tokenizer).reshape(-1)
        current_context_vector = context_vector(words, index, runtime, window_size)
        return best_reranked_word(
            word_vector,
            current_context_vector,
            vocab_embeddings,
            top_k,
            context_weight,
        )

    return replace_word
