from similarity_metrics import METRIC_LABELS


def print_settings(stopword_mode, heuristic_name, heuristic_config, vocabulary_size, metrics_config):
    """
    Print the active runtime settings.
    """

    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Stopword mode: {stopword_mode}")
    print(f"Heuristic: {heuristic_name}")
    for key, value in heuristic_config.items():
        if key != "name":
            print(f"Heuristic setting - {key}: {value}")

    enabled_metric_labels = [
        METRIC_LABELS[metric_name]
        for metric_name, enabled in metrics_config.items()
        if enabled
    ]
    print(f"Metrics: {', '.join(enabled_metric_labels)}")


def print_run_summary(summary):
    """
    Print summary statistics for a batch run.
    """

    print(f"Test sentences processed: {summary['sentence_count']}")
    for metric_name, metric_summary_data in summary["metrics"].items():
        metric_label = METRIC_LABELS[metric_name].lower()
        print(
            f"\t{metric_label} (average / median / p90): "
            f"{metric_summary_data['average']:.5f} / "
            f"{metric_summary_data['median']:.5f} / "
            f"{metric_summary_data['p90']:.5f}"
        )
