from sentence_rewriter import (
    find_closest_vocabulary_word as find_closest_word,
    load_test_sentences,
    main,
    process_test_sentences,
    rewrite_sentence as reconstruct_sentence,
)


__all__ = [
    "find_closest_word",
    "load_test_sentences",
    "main",
    "process_test_sentences",
    "reconstruct_sentence",
]


if __name__ == "__main__":
    main()
