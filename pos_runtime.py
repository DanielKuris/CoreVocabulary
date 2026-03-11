from collections import defaultdict

from nltk import pos_tag


UNIVERSAL_POS_TAGS = {
    "ADJ",
    "ADP",
    "ADV",
    "CONJ",
    "DET",
    "NOUN",
    "NUM",
    "PRT",
    "PRON",
    "VERB",
    ".",
    "X",
}
OPEN_CLASS_TAGS = {
    "ADJ",
    "ADV",
    "NOUN",
    "VERB",
}


def pos_tags_for_words(words):
    """
    Return universal POS tags for a sequence of words.
    """

    if not words:
        return []

    try:
        tagged_words = pos_tag(words, tagset="universal")
    except LookupError as error:
        raise RuntimeError(
            "The pos_aware heuristic requires NLTK POS resources. Install them with: "
            "python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('universal_tagset')\""
        ) from error

    return [tag for _, tag in tagged_words]


def token_pos_tags(tokens):
    """
    Return a token-indexed POS map for alphabetic tokens.
    """

    alphabetic_indices = [index for index, token in enumerate(tokens) if token.isalpha()]
    alphabetic_tokens = [tokens[index].lower() for index in alphabetic_indices]
    tagged_tokens = pos_tags_for_words(alphabetic_tokens)

    token_tags = {}
    for index, tag in zip(alphabetic_indices, tagged_tokens):
        token_tags[index] = tag

    return token_tags


def vocabulary_pos_groups(vocabulary):
    """
    Return vocabulary words grouped by universal POS tag.
    """

    grouped_vocabulary = defaultdict(list)
    tagged_vocabulary = pos_tags_for_words([word.lower() for word in vocabulary])

    for word, tag in zip(vocabulary, tagged_vocabulary):
        grouped_vocabulary[tag].append(word)

    return dict(grouped_vocabulary)


def candidate_vocabulary_words(index, token_tags, grouped_vocabulary, vocabulary):
    """
    Return the vocabulary candidates allowed for a token index.
    """

    token_tag = token_tags.get(index)
    if token_tag in grouped_vocabulary and grouped_vocabulary[token_tag]:
        return grouped_vocabulary[token_tag]

    if token_tag in OPEN_CLASS_TAGS:
        open_class_words = []
        for pos_tag_name in OPEN_CLASS_TAGS:
            open_class_words.extend(grouped_vocabulary.get(pos_tag_name, []))
        if open_class_words:
            return open_class_words

    return vocabulary
