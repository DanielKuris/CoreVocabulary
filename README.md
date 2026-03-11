# Core Vocabulary Sentence Rewriter

This repo takes a given core vocabulary, such as 600 words that are treated as the most important words in English, and tries to rewrite test sentences using only words from that vocabulary.

The goal is to map each test sentence onto the limited dictionary and then measure how similar the rewritten sentence remains to the original sentence.

## Files

- `sentence_rewriter.py`: loads the vocabulary and embeddings, rewrites the test sentences, and runs the batch workflow.
- `similarity_metrics.py`: computes sentence-level cosine similarity, exact Jaccard similarity, and semantic token overlap.
- `vocabulary_runtime.py`: loads vocabulary data and writes result summaries.
- `config_runtime.py`: loads and validates project settings.
- `output_runtime.py`: prints runtime settings and run summaries.
- `heuristics/nearest_word.py`: replaces each word with the nearest vocabulary word by cosine similarity.
- `heuristics/global_context.py`: replaces each word using its own embedding plus a weighted sentence-context vector.
- `heuristics/local_context.py`: replaces each word using its own embedding plus a weighted local-context vector from nearby words.
- `project_config.json`: project settings for the rewrite behavior and heuristic selection.
- `vocab_words_formatted.txt`: allowed vocabulary list.
- `vocab_embeddings_dict.pkl`: embeddings for the allowed vocabulary.
- `vocab600.csv`: saved vocabulary file.

## Config

`project_config.json` supports both general rewrite settings and heuristic-specific settings.

Example:

```json
{
  "stopword_mode": "vocab_only",
  "heuristic": {
    "name": "global_context",
    "global_context_weight": 0.15,
    "local_context_weight": 0.15,
    "local_context_window": 3
  }
}
```

### General settings

- `stopword_mode`
  - `preserve_original_stopwords`: keep stopwords from the original sentence even if they are not in the vocabulary
  - `vocab_only`: keep only words that are in the vocabulary, so stopwords outside the vocabulary are dropped

### Heuristics

- `heuristic.name`
  - `nearest_word`: compare each source word embedding directly to all vocabulary embeddings and choose the closest match
  - `global_context`: add a weighted sentence-context vector to each source word embedding before comparing it to the vocabulary
  - `local_context`: add a weighted local-context vector from nearby words before comparing the source word to the vocabulary

- `heuristic.global_context_weight`
  - used by `global_context`
  - controls how much the summed sentence embedding vector affects each replacement choice

- `heuristic.local_context_weight`
  - used by `local_context`
  - controls how much the nearby-word context vector affects each replacement choice

- `heuristic.local_context_window`
  - used by `local_context`
  - controls how many words before and after the current word are included in the local context
  - a value of `3` means up to three words before and three words after

## Metrics

- `Cosine Similarity`: compares sentence embeddings for the original and rewritten sentences. This is the main semantic similarity score.
- `Jaccard Similarity`: computes exact word overlap after lowercasing and keeping alphabetic words only. It is the size of the word-set intersection divided by the size of the word-set union.
- `Semantic Token Overlap`: compares token embeddings between the two sentences. For each token in the original sentence, it finds the most similar token in the rewritten sentence, sums those best-match similarities, and divides by the total number of tokens across both sentences. This is useful when the rewritten sentence uses different words that are still semantically close.

## Run

```bash
python sentence_rewriter.py
```

The script reads test sentences from `SimilarityTests/TestSentences.txt`, rewrites them using the provided vocabulary and the current config settings, compares the rewritten sentences to the originals, and writes the results to `SimilarityTests/TestResults.txt`.
