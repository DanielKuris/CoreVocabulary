import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from pos_runtime import candidate_vocabulary_words, token_pos_tags


def closest_vocabulary_word(target_vector, candidate_words, vocab_embeddings):
    """
    Return the nearest allowed vocabulary word for a target vector.
    """

    vocabulary_matrix = np.vstack([vocab_embeddings[word] for word in candidate_words])
    similarities = cosine_similarity(target_vector.reshape(1, -1), vocabulary_matrix)[0]
    closest_index = np.argmax(similarities)
    return candidate_words[closest_index]


def build_replacer(words, runtime, config):
    """
    Build a replacer that only searches vocabulary words with a matching POS tag.
    """

    vocab_embeddings = runtime["vocab_embeddings"]
    vocabulary = runtime["vocabulary"]
    vocabulary_pos_groups = runtime["vocabulary_pos_groups"]
    embed_word = runtime["embed_word"]
    model = runtime["model"]
    tokenizer = runtime["tokenizer"]
    sentence_pos_tags = token_pos_tags(words)

    def replace_word(word, index):
        """
        Replace a word using POS-aware nearest-neighbor lookup.
        """

        word_vector = embed_word(word, model, tokenizer).reshape(-1)
        candidate_words = candidate_vocabulary_words(
            index,
            sentence_pos_tags,
            vocabulary_pos_groups,
            vocabulary,
        )
        return closest_vocabulary_word(word_vector, candidate_words, vocab_embeddings)

    return replace_word
