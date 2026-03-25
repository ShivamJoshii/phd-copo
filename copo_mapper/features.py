from __future__ import annotations

from collections.abc import Iterable

ACTION_VERBS = {
    "remember": {"list", "define", "recall", "identify"},
    "understand": {"describe", "summarize", "explain", "classify"},
    "apply": {"apply", "use", "implement", "execute"},
    "analyze": {"analyze", "differentiate", "compare", "examine"},
    "evaluate": {"evaluate", "justify", "assess", "critique"},
    "create": {"design", "develop", "construct", "formulate"},
}

DOMAIN_TERMS = {
    "algorithms": {"algorithm", "complexity", "optimization"},
    "databases": {"database", "sql", "query", "relational"},
    "software": {"software", "system", "architecture", "deployment"},
    "communication": {"communicate", "report", "presentation", "write"},
    "ethics": {"ethics", "professional", "society", "impact"},
    "experiments": {"experiment", "measure", "analyze data", "hypothesis"},
}


BLOOM_ORDER = ["remember", "understand", "apply", "analyze", "evaluate", "create"]
BLOOM_INDEX = {level: i for i, level in enumerate(BLOOM_ORDER)}


def token_set(text: str) -> set[str]:
    return set(text.split())


def detect_bloom(tokens: Iterable[str]) -> str:
    token_values = set(tokens)
    for level in reversed(BLOOM_ORDER):
        if ACTION_VERBS[level] & token_values:
            return level
    return "understand"


def bloom_distance(level_a: str, level_b: str) -> int:
    return abs(BLOOM_INDEX[level_a] - BLOOM_INDEX[level_b])


def detect_domains(tokens: Iterable[str]) -> set[str]:
    token_values = set(tokens)
    matched = set()
    for domain, terms in DOMAIN_TERMS.items():
        if terms & token_values:
            matched.add(domain)
    return matched


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)
