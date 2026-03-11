from pathlib import Path
import pickle as pkl
import statistics

import numpy as np
from nltk.corpus import stopwords
from sklearn.preprocessing import normalize
from transformers import BertTokenizer, BertModel
import torch


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


def metric_summary(scores):
    """Return common summary statistics for a score list."""
    return {
        "average": statistics.mean(scores),
        "median": statistics.median(scores),
        "minimum": min(scores),
        "maximum": max(scores),
        "p90": float(np.percentile(scores, 90)),
    }


def summarize_similarity_results(results):
    """Return aggregate statistics for a batch of similarity results."""
    cosine_scores = [
        data["similarities"]["cosine_similarity_sentences_BERT"]
        for data in results.values()
    ]
    jaccard_scores = [
        data["similarities"]["jaccard_similarity"]
        for data in results.values()
    ]
    semantic_overlap_scores = [
        data["similarities"]["semantic_token_overlap"]
        for data in results.values()
    ]

    return {
        "sentence_count": len(results),
        "cosine": metric_summary(cosine_scores),
        "jaccard": metric_summary(jaccard_scores),
        "semantic_overlap": metric_summary(semantic_overlap_scores),
    }


def write_similarity_results(results, output_file=DEFAULT_RESULTS_PATH):
    """Write per-sentence scores and aggregate statistics to disk."""
    summary = summarize_similarity_results(results)

    with open(output_file, "w", encoding="utf-8") as file:
        for original, data in results.items():
            transformed = data["transformed"]
            cosine = data["similarities"]["cosine_similarity_sentences_BERT"]
            jaccard = data["similarities"]["jaccard_similarity"]
            semantic_overlap = data["similarities"]["semantic_token_overlap"]

            file.write(f"Original Sentence: {original}\n")
            file.write(f"Transformed Sentence: {transformed}\n")
            file.write(f"Cosine Similarity: {cosine:.5f}\n")
            file.write(f"Jaccard Similarity: {jaccard:.5f}\n")
            file.write(f"Semantic Token Overlap: {semantic_overlap:.5f}\n\n")

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
        file.write(f"90th Percentile: {summary['jaccard']['p90']:.5f}\n\n")

        file.write("Semantic Token Overlap:\n")
        file.write(f"Average: {summary['semantic_overlap']['average']:.5f}\n")
        file.write(f"Median: {summary['semantic_overlap']['median']:.5f}\n")
        file.write(f"Minimum: {summary['semantic_overlap']['minimum']:.5f}\n")
        file.write(f"Maximum: {summary['semantic_overlap']['maximum']:.5f}\n")
        file.write(f"90th Percentile: {summary['semantic_overlap']['p90']:.5f}\n")
