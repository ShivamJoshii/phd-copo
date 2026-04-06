from __future__ import annotations

import math
from collections import Counter
from functools import lru_cache


DEFAULT_SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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


@lru_cache(maxsize=2)
def _load_sbert_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def sbert_pair_similarity(
    co_texts: list[str],
    po_texts: list[str],
    model_name: str = DEFAULT_SBERT_MODEL,
) -> list[float] | None:
    if len(co_texts) != len(po_texts):
        raise ValueError("co_texts and po_texts must have the same length.")

    try:
        model = _load_sbert_model(model_name)
    except Exception:
        return None

    co_vectors = model.encode(co_texts, convert_to_tensor=True)
    po_vectors = model.encode(po_texts, convert_to_tensor=True)
    from sentence_transformers import util

    sims = util.cos_sim(co_vectors, po_vectors).diagonal()
    return [float(score) for score in sims]
