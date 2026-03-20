from pathlib import Path
import json


DEFAULT_CONFIG_PATH = Path("project_config.json")
DEFAULT_STOPWORD_MODE = "preserve_original_stopwords"
DEFAULT_HEURISTIC_NAME = "nearest_word"
DEFAULT_VOCABULARY_SIZE = 600
DEFAULT_METRICS = {
    "cosine_similarity_sentences_BERT": True,
    "jaccard_similarity": True,
    "semantic_token_overlap": True,
}
VALID_STOPWORD_MODES = {
    "preserve_original_stopwords",
    "vocab_only",
}
WEIGHT_KEYS = {
    "local_context_weight",
}


def load_project_config(path=DEFAULT_CONFIG_PATH):
    """
    Load project settings from disk.
    """

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def validate_weight(weight_name, value):
    """
    Validate that a heuristic weight is between 0 and 1.
    """

    numeric_value = float(value)
    if not 0.0 <= numeric_value <= 1.0:
        raise ValueError(f"{weight_name} must be between 0 and 1, got {value}")
    return numeric_value


def validate_vocabulary_size(value):
    """
    Validate that the configured vocabulary size matches an available CSV vocabulary.
    """

    vocabulary_size = int(value)
    if vocabulary_size < 100 or vocabulary_size > 2000 or vocabulary_size % 100 != 0:
        raise ValueError(
            f"vocabulary_size must be between 100 and 2000 in steps of 100, got {value}"
        )
    return vocabulary_size


def validate_local_context_window(value):
    """
    Validate that the local context window is -1 or a non-negative integer.
    """

    window_size = int(value)
    if window_size < -1:
        raise ValueError(
            f"local_context_window must be -1 or a non-negative integer, got {value}"
        )
    return window_size


def validate_top_k_candidates(value):
    """
    Validate that the top-k shortlist size is a positive integer.
    """

    top_k = int(value)
    if top_k < 1:
        raise ValueError(f"top_k_candidates must be a positive integer, got {value}")
    return top_k


def validate_reordering_max_tokens(value):
    """
    Validate that the permutation-search token limit is a positive integer.
    """

    max_tokens = int(value)
    if max_tokens < 1:
        raise ValueError(f"reordering_max_tokens must be a positive integer, got {value}")
    return max_tokens


def validate_metrics_config(metrics_config):
    """
    Validate and normalize the configured metric toggles.
    """

    normalized_metrics = DEFAULT_METRICS.copy()
    for metric_name, enabled in metrics_config.items():
        if metric_name not in DEFAULT_METRICS:
            raise ValueError(f"Unsupported metric: {metric_name}")
        if not isinstance(enabled, bool):
            raise ValueError(f"Metric toggle for {metric_name} must be true or false")
        normalized_metrics[metric_name] = enabled

    if not any(normalized_metrics.values()):
        raise ValueError("At least one metric must be enabled")

    return normalized_metrics


def get_stopword_mode(path=DEFAULT_CONFIG_PATH):
    """
    Return the configured stopword mode.
    """

    config = load_project_config(path)
    stopword_mode = config.get("stopword_mode", DEFAULT_STOPWORD_MODE)
    if stopword_mode not in VALID_STOPWORD_MODES:
        raise ValueError(f"Unsupported stopword_mode: {stopword_mode}")
    return stopword_mode


def get_vocabulary_size(path=DEFAULT_CONFIG_PATH):
    """
    Return the configured vocabulary size.
    """

    config = load_project_config(path)
    vocabulary_size = config.get("vocabulary_size", DEFAULT_VOCABULARY_SIZE)
    return validate_vocabulary_size(vocabulary_size)


def get_metrics_config(path=DEFAULT_CONFIG_PATH):
    """
    Return the configured metric toggles.
    """

    config = load_project_config(path)
    metrics_config = config.get("metrics", {})
    return validate_metrics_config(metrics_config)


def get_heuristic_config(path=DEFAULT_CONFIG_PATH):
    """
    Return the configured heuristic settings.
    """

    config = load_project_config(path)
    heuristic_config = config.get("heuristic", {})
    if "name" not in heuristic_config:
        heuristic_config["name"] = DEFAULT_HEURISTIC_NAME

    for key in WEIGHT_KEYS:
        if key in heuristic_config:
            heuristic_config[key] = validate_weight(key, heuristic_config[key])

    if "local_context_window" in heuristic_config:
        heuristic_config["local_context_window"] = validate_local_context_window(
            heuristic_config["local_context_window"]
        )

    if "top_k_candidates" in heuristic_config:
        heuristic_config["top_k_candidates"] = validate_top_k_candidates(
            heuristic_config["top_k_candidates"]
        )

    if "reordering_max_tokens" in heuristic_config:
        heuristic_config["reordering_max_tokens"] = validate_reordering_max_tokens(
            heuristic_config["reordering_max_tokens"]
        )

    return heuristic_config
