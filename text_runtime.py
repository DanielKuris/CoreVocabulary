import re

from nltk.corpus import stopwords


STOP_WORDS = set(stopwords.words("english"))
TOKEN_PATTERN = re.compile(r"[A-Za-z]+|[^\w\s]")


def tokenize_sentence(sentence):
    """
    Split a sentence into word and punctuation tokens.
    """

    return TOKEN_PATTERN.findall(sentence)


def join_tokens(tokens):
    """
    Join tokens into a sentence without adding spaces before punctuation.
    """

    sentence = ""
    for token in tokens:
        if not sentence:
            sentence = token
        elif re.fullmatch(r"[^\w\s]", token):
            sentence += token
        else:
            sentence += f" {token}"
    return sentence


def is_replaceable_word(word):
    """
    Return whether a word should be replaced rather than kept as-is.
    """

    return isinstance(word, str) and word.isalpha() and word.lower() not in STOP_WORDS
