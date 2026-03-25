from __future__ import annotations

import math
from collections import Counter


def _tf(text: str) -> Counter[str]:
    return Counter(text.split())


def _cosine(counter_a: Counter[str], counter_b: Counter[str]) -> float:
    dot = sum(counter_a[t] * counter_b.get(t, 0) for t in counter_a)
    norm_a = math.sqrt(sum(v * v for v in counter_a.values()))
    norm_b = math.sqrt(sum(v * v for v in counter_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def tfidf_pair_similarity(co_texts: list[str], po_texts: list[str]) -> list[float]:
    return [_cosine(_tf(co), _tf(po)) for co, po in zip(co_texts, po_texts, strict=True)]
