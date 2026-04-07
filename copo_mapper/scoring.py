from __future__ import annotations

from dataclasses import dataclass

from .features import bloom_distance, detect_bloom, detect_domains, jaccard, token_set


@dataclass(frozen=True)
class PairScore:
    score: int
    confidence: float
    explanation: str


def score_pair(co_text: str, po_text: str, semantic_similarity: float) -> PairScore:
    co_tokens = token_set(co_text)
    po_tokens = token_set(po_text)

    co_bloom = detect_bloom(co_tokens)
    po_bloom = detect_bloom(po_tokens)
    bloom_gap = bloom_distance(co_bloom, po_bloom)

    domain_overlap = jaccard(detect_domains(co_tokens), detect_domains(po_tokens))
    token_overlap = jaccard(co_tokens, po_tokens)

    composite = (
        0.45 * semantic_similarity
        + 0.2 * domain_overlap
        + 0.15 * token_overlap
        + 0.2 * max(0.0, 1 - (bloom_gap / 5))
    )

    if composite >= 0.50:
        label = 3
    elif composite >= 0.30:
        label = 2
    elif composite >= 0.10:
        label = 1
    else:
        label = 0

    explanation = (
        f"semantic={semantic_similarity:.2f}; domain_overlap={domain_overlap:.2f}; "
        f"token_overlap={token_overlap:.2f}; bloom={co_bloom}->{po_bloom} (gap={bloom_gap})"
    )
    return PairScore(score=label, confidence=round(composite, 3), explanation=explanation)
