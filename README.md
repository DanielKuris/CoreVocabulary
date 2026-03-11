# Core Vocabulary Sentence Rewriter

This repo takes a given core vocabulary, such as 600 words that are treated as the most important words in English, and tries to rewrite test sentences using only words from that vocabulary.

The goal is to map each test sentence onto the limited dictionary and then measure how similar the rewritten sentence remains to the original sentence.

## Files

- `sentence_rewriter.py`: loads the vocabulary and embeddings, rewrites the test sentences, writes results, and prints summary statistics.
- `similarity_metrics.py`: computes sentence-level cosine similarity and token-overlap similarity.
- `vocabulary_runtime.py`: loads vocabulary data, filters replaceable words, and writes result summaries.
- `vocab_words_formatted.txt`: allowed vocabulary list.
- `vocab_embeddings_dict.pkl`: embeddings for the allowed vocabulary.
- `vocab600.csv`: saved vocabulary file.

## Requirements

Install the packages in `requirements.txt`.

## Run

```bash
python sentence_rewriter.py
```

The script reads test sentences from `SimilarityTests/TestSentences.txt`, rewrites them using the provided vocabulary, compares the rewritten sentences to the originals, and writes the results to `SimilarityTests/TestResults.txt`.
