from pathlib import Path
import statistics

import numpy as np


DEFAULT_RESULTS_PATH = Path("SimilarityTests/TestResults.txt")


def metric_summary(scores):
    """
    Return common summary statistics for a score list.
    """

    return {
        "average": statistics.mean(scores),
        "median": statistics.median(scores),
        "minimum": min(scores),
        "maximum": max(scores),
        "p90": float(np.percentile(scores, 90)),
    }


def summarize_similarity_results(results):
    """
    Return aggregate statistics for a batch of similarity results.
    """

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
    """
    Write per-sentence scores and aggregate statistics to disk.
    """

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
