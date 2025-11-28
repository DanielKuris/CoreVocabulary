# Semantic Similarity and Vocabulary Filtering Toolkit

This project offers a set of tools for processing text data, transforming sentences, and assessing semantic similarity using a combination of BERT, GloVe, and other NLP models. It includes functions for vocabulary extraction, filtering, embedding generation, and sentence similarity measurement.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Scripts Overview](#scripts-overview)
4. [Examples](#examples)
5. [Dependencies](#dependencies)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites
Ensure you have the necessary packages:

```bash
torch
transformers
pandas
sentence-transformers
nltk
gensim
scikit-learn
```

### Additional Files
1. `glove.6B.100d.txt` — GloVe embeddings file, available at [GloVe site](https://nlp.stanford.edu/projects/glove/).
2. `SICK_sentences.csv` — sentence dataset for testing. (can be created by running `strip_sentences.py`)


## Usage

Follow these steps to use the toolkit:

1. **Generate the vocabulary**: 
   Run `vocabulary_semantic_filtering.py <enter size of vocabulary>`to create a vocabulary in size x

   ```bash
   python vocabulary_semantic_filtering.py <200(example)>
   ```


3. **Run similarity checks**:
    to compare original sentence to generated sentence using the current vocabulary.  This script uses `create_sentence.py` to transform sentences and saves a CSV file with the average cosine and Jaccard similarity scores.
   ```bash
   python `create_sentence.py` <"your sentence">
   ```
example result:
Original Sentence: the kids play in the garden
Transformed Sentence: kid playing porch
Cosine Similarity (Sentence-BERT): 0.8007192611694336
jaccard Similarity (Sentence-BERT): 0.4955784148640103

Original Sentence: Five kids are standing close together and none of the kids has a gun
Transformed Sentence: quarter kid stance occlude unite nix kid pistol
Cosine Similarity (Sentence-BERT): 0.22723878920078278
jaccard Similarity (Sentence-BERT): 0.39660113272459613




## Scripts Overview

### 1. `TransformersUtilityFunctions.py`

Defines the function `get_vec()` to obtain normalized sentence embeddings using a BERT-based model (`LaBSE`).

- **Function:** `get_vec(text_input: str) -> np.ndarray`
  - **Input:** String text to embed.
  - **Output:** Normalized embedding vector.

### 2. `check_similarity.py`

Reads a CSV file containing sentences, generates transformed sentences using `create_sentence`, and calculates similarity scores between original and transformed sentences using cosine and Jaccard similarity.

### 3. `create_sentence.py`

Transforms sentences by finding and substituting vocabulary terms with their closest semantic equivalents based on embeddings.

- **Main Functions:**
  - `preprocess_sentence()`
  - `find_closest_word()`
  - `transform_sentence()`

### 4. `sentence_similarity.py`

Contains helper functions for sentence similarity using BERT and Sentence-BERT embeddings. Provides both cosine and modified Jaccard similarity metrics.

- **Functions:**
  - `sentence_bert_embedding()`
  - `cosine_similarity_sentences()`
  - `jaccard_similarity_bert()`

### 5. `vocabulary_semantic_filtering.py`

Extracts a filtered vocabulary using WordNet and GloVe embeddings, with an option for additional filtering using contextual BERT embeddings.

- **Main Steps:**
  - Extract vocabulary from corpus
  - Filter with WordNet and GloVe embeddings
  - Refine with BERT embeddings for semantic relevance

## Examples

### Vocabulary Filtering and Embedding Generation

To generate embeddings for a filtered vocabulary list, use `get_vocab_dict.py`:

```bash
python get_vocab_dict.py
```

### Sentence Transformation and Similarity Checking

Transform a sentence and check similarity scores using `check_similarity.py`:

```bash
python check_similarity.py
```

## Dependencies

- Python 3.8+
- `torch`
- `transformers`
- `pandas`
- `sentence-transformers`
- `nltk`
- `gensim`
- `sklearn`
