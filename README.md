# Core Vocabulary Sentence Rewriter

This repo takes a given core vocabulary, such as 600 words that are treated as the most important words in English, and tries to rewrite test sentences using only words from that vocabulary.

The goal is to map each test sentence onto the limited dictionary and then measure how similar the rewritten sentence remains to the original sentence.

## Files

- `sentence_rewriter.py`: loads the vocabulary and embeddings, rewrites the test sentences, writes results, and prints summary statistics.
- `similarity_metrics.py`: computes sentence-level cosine similarity, exact Jaccard similarity, and semantic token overlap.
- `vocabulary_runtime.py`: loads vocabulary data, filters replaceable words, and writes result summaries.
- `vocab_words_formatted.txt`: allowed vocabulary list.
- `vocab_embeddings_dict.pkl`: embeddings for the allowed vocabulary.
- `vocab600.csv`: saved vocabulary file.

## Metrics

- `Cosine Similarity`: compares sentence embeddings for the original and rewritten sentences. This is the main semantic similarity score.
- `Jaccard Similarity`: computes exact word overlap after lowercasing and keeping alphabetic words only. It is the size of the word-set intersection divided by the size of the word-set union.
- `Semantic Token Overlap`: compares token embeddings between the two sentences. For each token in the original sentence, it finds the most similar token in the rewritten sentence, sums those best-match similarities, and divides by the total number of tokens across both sentences. This is useful when the rewritten sentence uses different words that are still semantically close.

## Requirements

Install the packages in `requirements.txt`.

## Run

```bash
python sentence_rewriter.py
```

The script reads test sentences from `SimilarityTests/TestSentences.txt`, rewrites them using the provided vocabulary, compares the rewritten sentences to the originals, and writes the results to `SimilarityTests/TestResults.txt`.
