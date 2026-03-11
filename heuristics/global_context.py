import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def closest_vocabulary_word(target_vector, vocab_embeddings):
    """Return the nearest vocabulary word for a target vector."""
    vocabulary_matrix = np.vstack(list(vocab_embeddings.values()))
    similarities = cosine_similarity(target_vector.reshape(1, -1), vocabulary_matrix)[0]
    closest_index = np.argmax(similarities)
    return list(vocab_embeddings.keys())[closest_index]


def sentence_context_vector(words, runtime):
    """Return the sum of word embeddings for alphabetic words in a sentence."""
    embed_word = runtime["embed_word"]
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]

    vectors = [
        embed_word(word.lower(), model, tokenizer).reshape(-1)
        for word in words
        if isinstance(word, str) and word.isalpha()
    ]

    if not vectors:
        sample_vector = embed_word("word", model, tokenizer).reshape(-1)
        return np.zeros_like(sample_vector)

    return np.sum(vectors, axis=0)


def build_replacer(words, runtime, config):
    """Build a replacer that adds weighted sentence context to each word vector."""
    vocab_embeddings = runtime["vocab_embeddings"]
    embed_word = runtime["embed_word"]
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    context_weight = float(config.get("global_context_weight", 0.0))
    context_vector = sentence_context_vector(words, runtime)

    def replace_word(word):
        word_vector = embed_word(word, model, tokenizer).reshape(-1)
        target_vector = word_vector + (context_weight * context_vector)
        return closest_vocabulary_word(target_vector, vocab_embeddings)

    return replace_word
