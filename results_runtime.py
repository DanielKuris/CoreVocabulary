from pathlib import Path
import statistics

import numpy as np

from similarity_metrics import METRIC_LABELS


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


def summarize_similarity_results(results, metrics_config):
    """
    Return aggregate statistics for the enabled similarity metrics.
    """

    summary = {
        "sentence_count": len(results),
        "metrics": {},
    }

    for metric_name, enabled in metrics_config.items():
        if not enabled:
            continue

        scores = [
            data["similarities"][metric_name]
            for data in results.values()
            if metric_name in data["similarities"]
        ]
        if scores:
            summary["metrics"][metric_name] = metric_summary(scores)

    return summary


def write_similarity_results(results, metrics_config, output_file=DEFAULT_RESULTS_PATH):
    """
    Write per-sentence scores and aggregate statistics to disk.
    """

    summary = summarize_similarity_results(results, metrics_config)

    with open(output_file, "w", encoding="utf-8") as file:
        for original, data in results.items():
            transformed = data["transformed"]

            file.write(f"Original Sentence: {original}\n")
            file.write(f"Transformed Sentence: {transformed}\n")
            for metric_name, enabled in metrics_config.items():
                if enabled and metric_name in data["similarities"]:
                    metric_label = METRIC_LABELS[metric_name]
                    metric_value = data["similarities"][metric_name]
                    file.write(f"{metric_label}: {metric_value:.5f}\n")
            file.write("\n")

        file.write(f"Test Sentences: {summary['sentence_count']}\n\n")

        for metric_name, metric_summary_data in summary["metrics"].items():
            metric_label = METRIC_LABELS[metric_name]
            file.write(f"{metric_label}:\n")
            file.write(f"Average: {metric_summary_data['average']:.5f}\n")
            file.write(f"Median: {metric_summary_data['median']:.5f}\n")
            file.write(f"Minimum: {metric_summary_data['minimum']:.5f}\n")
            file.write(f"Maximum: {metric_summary_data['maximum']:.5f}\n")
            file.write(f"90th Percentile: {metric_summary_data['p90']:.5f}\n\n")
