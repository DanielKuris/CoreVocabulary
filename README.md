# Core Vocabulary Sentence Rewriter

This repo takes a given core vocabulary, such as 600 words that are treated as the most important words in English, and tries to rewrite test sentences using only words from that vocabulary.

The goal is to map each test sentence onto the limited dictionary and then measure how similar the rewritten sentence remains to the original sentence.

## Files

- `sentence_rewriter.py`: loads the configured CSV vocabulary, rewrites the test sentences, and runs the batch workflow.
- `similarity_metrics.py`: computes sentence-level cosine similarity, exact Jaccard similarity, and semantic token overlap.
- `vocabulary_runtime.py`: loads vocabulary data from `vocabularies/csv_vocab` and builds LaBSE vocabulary embeddings at startup.
- `text_runtime.py`: tokenizes sentences, rejoins tokens, and decides which words are replaceable.
- `results_runtime.py`: summarizes similarity scores and writes the batch output file.
- `config_runtime.py`: loads and validates project settings.
- `output_runtime.py`: prints runtime settings and run summaries.
- `heuristics/nearest_word.py`: replaces each word with the nearest vocabulary word by cosine similarity.
- `heuristics/local_context.py`: replaces each word using a weighted blend of the word embedding and a context vector from nearby words or the whole sentence.
- `heuristics/top_k_local_context.py`: reranks the top-k nearest vocabulary candidates using local or sentence-level context.
- `project_config.json`: project settings for vocabulary size, rewrite behavior, metric toggles, and heuristic selection.
- `vocabularies/csv_vocab/`: CSV vocabularies from 100 to 2000 words.

## Config

`project_config.json` supports general rewrite settings, vocabulary selection, metric toggles, and heuristic-specific settings.

Example:

```json
{
  "vocabulary_size": 600,
  "stopword_mode": "vocab_only",
  "metrics": {
    "cosine_similarity_sentences_BERT": true,
    "jaccard_similarity": true,
    "semantic_token_overlap": true
  },
  "heuristic": {
    "name": "top_k_local_context",
    "local_context_weight": 0.15,
    "local_context_window": 3,
    "top_k_candidates": 5
  }
}
```

### General settings

- `vocabulary_size`
  - selects `vocabularies/csv_vocab/vocab{size}.csv`
  - valid values are `100` through `2000` in steps of `100`

- `stopword_mode`
  - `preserve_original_stopwords`: keep stopwords from the original sentence even if they are not in the vocabulary
  - `vocab_only`: keep only words that are in the vocabulary, so stopwords outside the vocabulary are dropped

### Metrics

- `metrics.cosine_similarity_sentences_BERT`
  - enables or disables sentence-level cosine similarity

- `metrics.jaccard_similarity`
  - enables or disables exact word-set Jaccard similarity

- `metrics.semantic_token_overlap`
  - enables or disables semantic token overlap

At least one metric must be enabled.

### Heuristics

- `heuristic.name`
  - `nearest_word`: compare each source word embedding directly to all vocabulary embeddings and choose the closest match
  - `local_context`: compare each source word using `(1 - local_context_weight) * word_vector + local_context_weight * context_vector`, where `context_vector` is built from context words around the current word
  - `top_k_local_context`: take the top `k` nearest vocabulary words for the source word, then rerank only that shortlist using the context vector

- `heuristic.local_context_weight`
  - used by `local_context` and `top_k_local_context`
  - controls the blend between the current word embedding signal and the context signal
  - `0` means use only the word similarity signal
  - `1` means use only the context signal

- `heuristic.local_context_window`
  - used by `local_context` and `top_k_local_context`
  - controls how many words before and after the current word are included in the context
  - for example: `3` means up to three words before and three words after
  - `-1` means use the whole sentence as context

- `heuristic.top_k_candidates`
  - used by `top_k_local_context`
  - controls how many nearest vocabulary candidates are shortlisted before reranking
  - for example: `5` means rerank the five nearest vocabulary words

## Metrics

- `Cosine Similarity`: compares sentence embeddings for the original and rewritten sentences. This is the main semantic similarity score.
- `Jaccard Similarity`: computes exact word overlap after lowercasing and keeping alphabetic words only. It is the size of the word-set intersection divided by the size of the word-set union.
- `Semantic Token Overlap`: compares token embeddings between the two sentences. For each token in the original sentence, it finds the most similar token in the rewritten sentence, sums those best-match similarities, and divides by the total number of tokens across both sentences. This is useful when the rewritten sentence uses different words that are still semantically close.

## Embeddings

Both sides of the replacement lookup use `setu4993/LaBSE`.

- The vocabulary words are read from `vocabularies/csv_vocab/vocab{size}.csv`.
- At the start of the run, every vocabulary word is embedded with LaBSE.
- During rewriting, each replaceable source word is also embedded with LaBSE.
- Replacement is done by comparing those LaBSE vectors directly.

## Run

```bash
python sentence_rewriter.py
```

The script reads test sentences from `SimilarityTests/TestSentences.txt`, rewrites them using the configured CSV vocabulary and the current config settings, compares the rewritten sentences to the originals, and writes the results to `SimilarityTests/TestResults.txt`.
