from heuristics.global_context import build_replacer as build_global_context_replacer
from heuristics.nearest_word import build_replacer as build_nearest_word_replacer


HEURISTIC_BUILDERS = {
    "nearest_word": build_nearest_word_replacer,
    "global_context": build_global_context_replacer,
}


def get_heuristic_builder(name):
    """Return the builder function for a configured heuristic."""
    if name not in HEURISTIC_BUILDERS:
        raise ValueError(f"Unsupported heuristic: {name}")
    return HEURISTIC_BUILDERS[name]
