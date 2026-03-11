def print_settings(stopword_mode, heuristic_name, heuristic_config, vocabulary_size):
    """
    Print the active runtime settings.
    """

    print(f"Vocabulary size: {vocabulary_size}")
    print(f"Stopword mode: {stopword_mode}")
    print(f"Heuristic: {heuristic_name}")
    for key, value in heuristic_config.items():
        if key != "name":
            print(f"Heuristic setting - {key}: {value}")


def print_run_summary(summary):
    """
    Print summary statistics for a batch run.
    """

    print(f"Test sentences processed: {summary['sentence_count']}")
    print(
        "Cosine similarity (average / median / p90): "
        f"{summary['cosine']['average']:.5f} / "
        f"{summary['cosine']['median']:.5f} / "
        f"{summary['cosine']['p90']:.5f}"
    )
    print(
        "Jaccard similarity (average / median / p90): "
        f"{summary['jaccard']['average']:.5f} / "
        f"{summary['jaccard']['median']:.5f} / "
        f"{summary['jaccard']['p90']:.5f}"
    )
    print(
        "Semantic token overlap (average / median / p90): "
        f"{summary['semantic_overlap']['average']:.5f} / "
        f"{summary['semantic_overlap']['median']:.5f} / "
        f"{summary['semantic_overlap']['p90']:.5f}"
    )
