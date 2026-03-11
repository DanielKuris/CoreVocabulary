from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from vocabulary_runtime import (
    compare_original_and_transformed,
    embed_word,
    filter_replaceable_words,
    find_out_of_vocabulary_words,
    load_replacement_model,
    load_vocab_embeddings,
    load_vocabulary,
    summarize_similarity_results,
    write_similarity_results,
)


DEFAULT_INPUT_PATH = Path("SimilarityTests/TestSentences.txt")
DEFAULT_OUTPUT_PATH = Path("SimilarityTests/TestResults.txt")
REPLACEMENT_MODEL, REPLACEMENT_TOKENIZER = load_replacement_model()
VOCABULARY = load_vocabulary()
VOCAB_EMBEDDINGS = load_vocab_embeddings()


def find_closest_vocabulary_word(word):
    """Return the nearest allowed-vocabulary word for a source word."""
    word_embedding = embed_word(word, REPLACEMENT_MODEL, REPLACEMENT_TOKENIZER)
    word_vector = word_embedding.reshape(1, -1)
    vocabulary_matrix = np.vstack(list(VOCAB_EMBEDDINGS.values()))
    similarities = cosine_similarity(word_vector, vocabulary_matrix)[0]
    closest_index = np.argmax(similarities)
    return list(VOCAB_EMBEDDINGS.keys())[closest_index]


def rewrite_sentence(sentence):
    """Return replacement words for out-of-vocabulary tokens in a sentence."""
    words_to_replace = find_out_of_vocabulary_words(sentence, VOCABULARY)
    words_to_replace = filter_replaceable_words(words_to_replace)

    substitutions = {}
    for word in words_to_replace:
        closest_word = find_closest_vocabulary_word(word)
        if closest_word:
            substitutions[word] = closest_word

    transformed_words = []
    for word in words_to_replace:
        transformed_words.append(substitutions[word])

    return " ".join(transformed_words)


def load_test_sentences(input_path=DEFAULT_INPUT_PATH):
    """Read test sentences from disk."""
    sentences = []
    with open(input_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                sentence = line.split(". ", 1)[1] if ". " in line else line
                sentences.append(sentence)
    return sentences


def process_test_sentences(input_path=DEFAULT_INPUT_PATH, output_path=DEFAULT_OUTPUT_PATH):
    """Rewrite test sentences and save similarity scores."""
    results = {}
    for sentence in load_test_sentences(input_path):
        transformed = rewrite_sentence(sentence)
        similarities = compare_original_and_transformed(sentence, transformed)
        results[sentence] = {
            "transformed": transformed,
            "similarities": similarities,
        }

    write_similarity_results(results, output_path)
    return results


def print_run_summary(results):
    """Print summary statistics for a batch run."""
    summary = summarize_similarity_results(results)
    print(f"Test sentences processed: {summary['sentence_count']}")
    print(f"Average cosine similarity: {summary['cosine']['average']:.5f}")
    print(f"Median cosine similarity: {summary['cosine']['median']:.5f}")
    print(f"Cosine similarity 90th percentile: {summary['cosine']['p90']:.5f}")
    print(f"Average Jaccard similarity: {summary['jaccard']['average']:.5f}")
    print(f"Median Jaccard similarity: {summary['jaccard']['median']:.5f}")
    print(f"Jaccard similarity 90th percentile: {summary['jaccard']['p90']:.5f}")


def main():
    print(f"Processing sentences from {DEFAULT_INPUT_PATH} ...")
    results = process_test_sentences()
    print_run_summary(results)
    print(f"Processing complete. Results written to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
