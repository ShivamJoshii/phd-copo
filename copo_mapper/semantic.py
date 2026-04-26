from __future__ import annotations

import math
from collections import Counter
from typing import Optional


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
    if len(co_texts) != len(po_texts):
        raise ValueError("co_texts and po_texts must have the same length.")
    return [_cosine(_tf(co), _tf(po)) for co, po in zip(co_texts, po_texts, strict=True)]


def sbert_pair_similarity(
    co_texts: list[str],
    po_texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Optional[list[float]]:
    """
    Compute cosine similarity using Sentence-BERT embeddings.

    Returns None if sentence-transformers is unavailable in the runtime.
    """
    if len(co_texts) != len(po_texts):
        raise ValueError("co_texts and po_texts must have the same length.")
    if not co_texts:
        return []

    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        return None

    model = SentenceTransformer(model_name)
    co_embeddings = model.encode(co_texts, convert_to_numpy=True, normalize_embeddings=True)
    po_embeddings = model.encode(po_texts, convert_to_numpy=True, normalize_embeddings=True)

    return [float((co_embeddings[i] * po_embeddings[i]).sum()) for i in range(len(co_texts))]
