from pathlib import Path
import json


DEFAULT_CONFIG_PATH = Path("project_config.json")
DEFAULT_STOPWORD_MODE = "preserve_original_stopwords"
DEFAULT_HEURISTIC_NAME = "nearest_word"
VALID_STOPWORD_MODES = {
    "preserve_original_stopwords",
    "vocab_only",
}


def load_project_config(path=DEFAULT_CONFIG_PATH):
    """Load project settings from disk."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def get_stopword_mode(path=DEFAULT_CONFIG_PATH):
    """Return the configured stopword mode."""
    config = load_project_config(path)
    stopword_mode = config.get("stopword_mode", DEFAULT_STOPWORD_MODE)
    if stopword_mode not in VALID_STOPWORD_MODES:
        raise ValueError(f"Unsupported stopword_mode: {stopword_mode}")
    return stopword_mode


def get_heuristic_config(path=DEFAULT_CONFIG_PATH):
    """Return the configured heuristic settings."""
    config = load_project_config(path)
    heuristic_config = config.get("heuristic", {})
    if "name" not in heuristic_config:
        heuristic_config["name"] = DEFAULT_HEURISTIC_NAME
    return heuristic_config
