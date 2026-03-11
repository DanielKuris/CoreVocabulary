import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def closest_vocabulary_word(target_vector, vocab_embeddings):
    """
    Return the nearest vocabulary word for a target vector.
    """

    vocabulary_matrix = np.vstack(list(vocab_embeddings.values()))
    similarities = cosine_similarity(target_vector.reshape(1, -1), vocabulary_matrix)[0]
    closest_index = np.argmax(similarities)
    return list(vocab_embeddings.keys())[closest_index]


def context_vector(words, index, runtime, window_size):
    """
    Return the average embedding of context words around the current word.
    """

    embed_word = runtime["embed_word"]
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]

    if window_size == -1:
        start = 0
        end = len(words)
    else:
        start = max(0, index - window_size)
        end = min(len(words), index + window_size + 1)

    vectors = []
    for neighbor_index in range(start, end):
        if neighbor_index == index:
            continue
        neighbor = words[neighbor_index]
        if isinstance(neighbor, str) and neighbor.isalpha():
            vectors.append(embed_word(neighbor.lower(), model, tokenizer).reshape(-1))

    if not vectors:
        sample_vector = embed_word("word", model, tokenizer).reshape(-1)
        return np.zeros_like(sample_vector)

    return np.mean(vectors, axis=0)


def build_replacer(words, runtime, config):
    """
    Build a replacer that blends each word vector with its context vector.
    """

    vocab_embeddings = runtime["vocab_embeddings"]
    embed_word = runtime["embed_word"]
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    context_weight = float(config.get("local_context_weight", 0.0))
    window_size = int(config.get("local_context_window", 3))

    def replace_word(word, index):
        """
        Replace a word using context-aware nearest-neighbor lookup.
        """

        word_vector = embed_word(word, model, tokenizer).reshape(-1)
        current_context_vector = context_vector(words, index, runtime, window_size)
        target_vector = ((1 - context_weight) * word_vector) + (context_weight * current_context_vector)
        return closest_vocabulary_word(target_vector, vocab_embeddings)

    return replace_word
