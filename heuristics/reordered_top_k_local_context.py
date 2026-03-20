import itertools

from heuristics.top_k_local_context import build_replacer as build_top_k_local_context_replacer
from similarity_metrics import cosine_similarity_sentences, sentence_bert_embedding
from text_runtime import is_replaceable_word, join_tokens


DEFAULT_REORDERING_MAX_TOKENS = 6


def reordered_sentence(words, replaceable_indices, reordered_words):
    """
    Return a sentence with reordered replacement words filled into their slots.
    """

    candidate_tokens = list(words)
    for index, replacement_word in zip(replaceable_indices, reordered_words):
        candidate_tokens[index] = replacement_word
    return join_tokens(candidate_tokens)


def best_reordered_replacements(words, replaceable_indices, base_replacements, max_tokens):
    """
    Return the replacement order with the best sentence-level cosine similarity.
    """

    if len(base_replacements) <= 1 or len(base_replacements) > max_tokens:
        return base_replacements

    original_sentence = join_tokens(words)
    original_embedding = sentence_bert_embedding(original_sentence)
    best_order = base_replacements
    best_score = float("-inf")

    for candidate_order in itertools.permutations(base_replacements):
        candidate_sentence = reordered_sentence(words, replaceable_indices, candidate_order)
        candidate_embedding = sentence_bert_embedding(candidate_sentence)
        candidate_score = cosine_similarity_sentences(original_embedding, candidate_embedding)

        if candidate_score > best_score:
            best_score = candidate_score
            best_order = list(candidate_order)

    return best_order


def build_replacer(words, runtime, config):
    """
    Build a replacer that tries reordering replacement words to improve sentence similarity.
    """

    base_replacer = build_top_k_local_context_replacer(words, runtime, config)
    max_tokens = int(config.get("reordering_max_tokens", DEFAULT_REORDERING_MAX_TOKENS))
    replaceable_indices = [
        index
        for index, word in enumerate(words)
        if is_replaceable_word(word)
    ]
    base_replacements = [
        base_replacer(words[index].lower(), index)
        for index in replaceable_indices
    ]
    reordered_replacements = best_reordered_replacements(
        words,
        replaceable_indices,
        base_replacements,
        max_tokens,
    )
    replacement_map = {
        index: replacement_word
        for index, replacement_word in zip(replaceable_indices, reordered_replacements)
    }

    def replace_word(word, index):
        """
        Return the precomputed replacement word for a token index.
        """

        return replacement_map[index]

    return replace_word
