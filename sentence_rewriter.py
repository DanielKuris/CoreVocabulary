from pathlib import Path

from config_runtime import (
    get_heuristic_config,
    get_metrics_config,
    get_stopword_mode,
    get_vocabulary_size,
)
from heuristics import get_heuristic_builder
from output_runtime import print_run_summary, print_settings
from results_runtime import summarize_similarity_results, write_similarity_results
from similarity_metrics import compare_sentences
from text_runtime import is_replaceable_word, join_tokens, tokenize_sentence
from vocabulary_runtime import (
    build_vocab_embeddings,
    embed_word,
    load_replacement_model,
    load_vocabulary,
)


DEFAULT_INPUT_PATH = Path("SimilarityTests/TestSentences.txt")
DEFAULT_OUTPUT_PATH = Path("SimilarityTests/TestResults.txt")
MODEL, TOKENIZER = load_replacement_model()
HEURISTIC_CONFIG = get_heuristic_config()
HEURISTIC_NAME = HEURISTIC_CONFIG["name"]
METRICS_CONFIG = get_metrics_config()
STOPWORD_MODE = get_stopword_mode()
VOCABULARY_SIZE = get_vocabulary_size()
VOCABULARY = load_vocabulary(VOCABULARY_SIZE)
VOCAB_EMBEDDINGS = build_vocab_embeddings(VOCABULARY, MODEL, TOKENIZER)
RUNTIME = {
    "embed_word": embed_word,
    "model": MODEL,
    "tokenizer": TOKENIZER,
    "vocab_embeddings": VOCAB_EMBEDDINGS,
}


def keep_original_word(word):
    """
    Return whether a word should stay unchanged in the rewritten sentence.
    """

    normalized_word = word.lower()
    if normalized_word in VOCABULARY:
        return True
    if STOPWORD_MODE == "preserve_original_stopwords" and not is_replaceable_word(word):
        return True
    return False


def rewrite_sentence(sentence):
    """
    Return a sentence with out-of-vocabulary content words replaced.
    """

    words = tokenize_sentence(sentence)
    replace_word = get_heuristic_builder(HEURISTIC_NAME)(words, RUNTIME, HEURISTIC_CONFIG)

    rewritten_words = []
    for index, word in enumerate(words):
        normalized_word = word.lower()
        if keep_original_word(word):
            rewritten_words.append(word)
        elif is_replaceable_word(word):
            rewritten_words.append(replace_word(normalized_word, index))

    return join_tokens(rewritten_words)


def load_test_sentences(input_path=DEFAULT_INPUT_PATH):
    """
    Read test sentences from disk.
    """

    sentences = []
    with open(input_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                sentence = line.split(". ", 1)[1] if ". " in line else line
                sentences.append(sentence)
    return sentences


def process_test_sentences(input_path=DEFAULT_INPUT_PATH, output_path=DEFAULT_OUTPUT_PATH):
    """
    Rewrite test sentences and save similarity scores.
    """

    results = {}
    for sentence in load_test_sentences(input_path):
        transformed = rewrite_sentence(sentence)
        results[sentence] = {
            "transformed": transformed,
            "similarities": compare_sentences(sentence, transformed, METRICS_CONFIG),
        }

    write_similarity_results(results, METRICS_CONFIG, output_path)
    return results


def main():
    """
    Run the batch rewrite workflow for the test sentence file.
    """

    print(f"Processing sentences from {DEFAULT_INPUT_PATH} ...")
    print_settings(STOPWORD_MODE, HEURISTIC_NAME, HEURISTIC_CONFIG, VOCABULARY_SIZE, METRICS_CONFIG)
    results = process_test_sentences()
    print_run_summary(summarize_similarity_results(results, METRICS_CONFIG))
    print(f"Processing complete. Results written to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
