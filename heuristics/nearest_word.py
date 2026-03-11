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


def build_replacer(words, runtime, config):
    """
    Build the baseline nearest-word replacer for a sentence.
    """

    vocab_embeddings = runtime["vocab_embeddings"]
    embed_word = runtime["embed_word"]
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]

    def replace_word(word, index):
        """
        Replace a word using direct nearest-neighbor lookup.
        """

        word_vector = embed_word(word, model, tokenizer).reshape(-1)
        return closest_vocabulary_word(word_vector, vocab_embeddings)

    return replace_word
