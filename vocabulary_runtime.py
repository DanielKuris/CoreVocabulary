from pathlib import Path
import pickle as pkl
import statistics

from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from transformers import BertTokenizer, BertModel
import numpy as np
import torch

from similarity_metrics import compare_sentences


DEFAULT_VOCAB_PATH = Path("vocab_words_formatted.txt")
DEFAULT_VOCAB_EMBEDDINGS_PATH = Path("vocab_embeddings_dict.pkl")
DEFAULT_RESULTS_PATH = Path("SimilarityTests/TestResults.txt")


def embed_word(word, model, tokenizer):
    """Return a normalized embedding for a single word."""
    input_ids = torch.tensor(tokenizer.encode(word)).unsqueeze(0)
    outputs = model(input_ids)
    embedding = outputs[1].detach().numpy()
    return normalize(embedding)


def load_vocab_embeddings(path=DEFAULT_VOCAB_EMBEDDINGS_PATH):
    """Load allowed-vocabulary embeddings from disk."""
    with open(path, "rb") as file:
        return pkl.load(file)


def load_vocabulary(path=DEFAULT_VOCAB_PATH):
    """Load the allowed vocabulary list from disk."""
    with open(path, "r", encoding="utf-8") as file:
        return eval(file.read())


def load_replacement_model():
    """Load the word-level model used for nearest-vocabulary replacement."""
    tokenizer = BertTokenizer.from_pretrained("setu4993/LaBSE")
    model = BertModel.from_pretrained("setu4993/LaBSE")
    return model, tokenizer


def compare_original_and_transformed(original_sentence, transformed_sentence):
    """Return similarity scores for two sentences."""
    return compare_sentences(original_sentence, transformed_sentence)


def filter_replaceable_words(words):
    """Keep alphabetic, non-stopword tokens and lowercase them."""
    stop_words = set(stopwords.words("english"))
    return [
        word.lower()
        for word in words
        if isinstance(word, str) and word.isalpha() and word.lower() not in stop_words
    ]


def find_out_of_vocabulary_words(sentence, vocabulary):
    """Return tokens that are not part of the allowed vocabulary."""
    return [word for word in sentence.split() if word not in vocabulary]


def load_glove_vocab_embeddings(vocabulary, glove_file="glove.6B.100d.txt"):
    """Return GloVe embeddings for vocabulary words present in the model."""
    model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
    try:
        vocab_embeddings = {word: model[word] for word in vocabulary if word in model}
        if not vocab_embeddings:
            raise ValueError("no words from vocabulary found in model")
    except Exception:
        print("no words found from vocabulary found in model")
        return {}
    return vocab_embeddings


def summarize_similarity_results(results):
    """Return aggregate statistics for a batch of similarity results."""
    cosine_scores = [
        data["similarities"]["cosine_similarity_sentences_BERT"]
        for data in results.values()
    ]
    jaccard_scores = [
        data["similarities"]["jaccard_similarity_BERT"]
        for data in results.values()
    ]

    return {
        "sentence_count": len(results),
        "cosine": {
            "average": statistics.mean(cosine_scores),
            "median": statistics.median(cosine_scores),
            "minimum": min(cosine_scores),
            "maximum": max(cosine_scores),
            "p90": float(np.percentile(cosine_scores, 90)),
        },
        "jaccard": {
            "average": statistics.mean(jaccard_scores),
            "median": statistics.median(jaccard_scores),
            "minimum": min(jaccard_scores),
            "maximum": max(jaccard_scores),
            "p90": float(np.percentile(jaccard_scores, 90)),
        },
    }


def write_similarity_results(results, output_file=DEFAULT_RESULTS_PATH):
    """Write per-sentence scores and aggregate statistics to disk."""
    summary = summarize_similarity_results(results)

    with open(output_file, "w", encoding="utf-8") as file:
        for original, data in results.items():
            transformed = data["transformed"]
            cosine = data["similarities"]["cosine_similarity_sentences_BERT"]
            jaccard = data["similarities"]["jaccard_similarity_BERT"]

            file.write(f"Original Sentence: {original}\n")
            file.write(f"Transformed Sentence: {transformed}\n")
            file.write(f"Cosine Similarity: {cosine:.5f}\n")
            file.write(f"Jaccard Similarity: {jaccard:.5f}\n\n")

        file.write(f"Test Sentences: {summary['sentence_count']}\n\n")

        file.write("Cosine Similarity:\n")
        file.write(f"Average: {summary['cosine']['average']:.5f}\n")
        file.write(f"Median: {summary['cosine']['median']:.5f}\n")
        file.write(f"Minimum: {summary['cosine']['minimum']:.5f}\n")
        file.write(f"Maximum: {summary['cosine']['maximum']:.5f}\n")
        file.write(f"90th Percentile: {summary['cosine']['p90']:.5f}\n\n")

        file.write("Jaccard Similarity:\n")
        file.write(f"Average: {summary['jaccard']['average']:.5f}\n")
        file.write(f"Median: {summary['jaccard']['median']:.5f}\n")
        file.write(f"Minimum: {summary['jaccard']['minimum']:.5f}\n")
        file.write(f"Maximum: {summary['jaccard']['maximum']:.5f}\n")
        file.write(f"90th Percentile: {summary['jaccard']['p90']:.5f}\n")
