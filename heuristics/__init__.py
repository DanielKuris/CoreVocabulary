from heuristics.local_context import build_replacer as build_local_context_replacer
from heuristics.nearest_word import build_replacer as build_nearest_word_replacer
from heuristics.pos_aware import build_replacer as build_pos_aware_replacer
from heuristics.reordered_top_k_local_context import build_replacer as build_reordered_top_k_local_context_replacer
from heuristics.top_k_local_context import build_replacer as build_top_k_local_context_replacer


HEURISTIC_BUILDERS = {
    "nearest_word": build_nearest_word_replacer,
    "local_context": build_local_context_replacer,
    "top_k_local_context": build_top_k_local_context_replacer,
    "reordered_top_k_local_context": build_reordered_top_k_local_context_replacer,
    "pos_aware": build_pos_aware_replacer,
}


def get_heuristic_builder(name):
    """
    Return the builder function for a configured heuristic.
    """

    if name not in HEURISTIC_BUILDERS:
        raise ValueError(f"Unsupported heuristic: {name}")
    return HEURISTIC_BUILDERS[name]
