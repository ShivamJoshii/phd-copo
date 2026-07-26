from dataclasses import dataclass

from .features import (
    GENERIC_DOMAINS,
    bloom_distance,
    detect_bloom,
    detect_domains,
    jaccard,
    token_set,
)

# Composite-score cutoffs (t3, t2, t1) per semantic backend: label 3 if
# composite >= t3, 2 if >= t2, 1 if >= t1, else 0.
#
# The composite is 0.45 * semantic_similarity + up to 0.55 from features
# (0.2 domain overlap + 0.15 token overlap + 0.2 bloom-proximity).
#
# Bloom term for unrelated texts (measured on the real MBA exports, 1980
# CO x PO pairs, leading-verb detection): with the expanded verb lists most
# outcome statements DO carry a detectable Bloom level (levels spread across
# the whole taxonomy: understand 43%, create 21%, apply 21%, evaluate 7%,
# analyze 6%, remember 1% of real COs), so unrelated pairs usually have a
# nonzero gap. The bloom term averages ~0.13 (median 0.12; only 20% of real
# pairs sit at gap 0). The 0.2 worst case still exists — a pair whose texts
# both contain NO action verb at all defaults both sides to "understand"
# (gap 0) — so threshold anchors below conservatively use the 0.2 floor, and
# typical real unrelated pairs land ~0.07 lower.
#
# Domain term for unrelated texts (same measurement): 0 for 93% of real
# pairs, mean 0.007, > 0.1 for only 1.3% of pairs. Catch-all management
# vocabulary was pruned in features.DOMAIN_TERMS and a single shared
# GENERIC domain earns half credit (see score_pair), so the domain term no
# longer inflates unrelated same-discipline pairs (pre-fix it pushed 40%+
# of a real management grid to label 2 at semantic similarity 0).
#
# tfidf: TF-IDF cosine is near 0 for loosely related texts, so the original
#   cutoffs (0.50, 0.30, 0.10) remain valid and are kept exactly for
#   backwards compatibility. Measured on the real exports these cutoffs now
#   give 0:26% / 1:72% / 2:1.7% / 3:0.2% program-wide, with every label-3
#   pair a genuinely strong match (e.g. "develop leadership qualities" ->
#   "develop Value based Leadership ability").
#
# sbert (all-MiniLM-L6-v2): cosine sits ~0.2-0.5 even for barely related
#   sentence pairs (unrelated < 0.25, weak ~0.25-0.45, moderate ~0.45-0.6,
#   strong > 0.6). Anchor arithmetic (bloom shown at its verb-less
#   worst-case 0.20; typical real unrelated pairs carry ~0.13):
#     unrelated: sim 0.20, no overlap        -> 0.45*0.20 + 0.20        = 0.29 -> 0
#     weak:      sim 0.35, token jac ~0.1    -> 0.158 + 0.015 + 0.20    = 0.37 -> 1
#     moderate:  sim 0.52, domain jac 0.5,
#                token jac ~0.15             -> 0.234 + 0.10 + 0.023 + 0.20 = 0.56 -> 2
#     strong:    sim 0.65, domain jac 0.5,
#                token jac ~0.3              -> 0.293 + 0.10 + 0.045 + 0.20 = 0.64 -> 3
#   Hence (0.62, 0.50, 0.33): 0.29 < 0.33 < 0.37 < 0.50 <= 0.56 < 0.62 <= 0.64.
#
# bert (bert-base-uncased, mean-pooled): anisotropic embeddings push cosine
#   even higher (unrelated ~0.5-0.6, related ~0.7-0.9). Anchor arithmetic
#   (same worst-case bloom convention):
#     unrelated: sim 0.55, no overlap        -> 0.248 + 0.20            = 0.45 -> 0
#     weak:      sim 0.65                    -> 0.293 + 0.20            = 0.49 -> 1
#     moderate:  sim 0.75, domain jac 0.5,
#                token jac ~0.15             -> 0.338 + 0.10 + 0.023 + 0.20 = 0.66 -> 2
#     strong:    sim 0.85, domain jac 0.5,
#                token jac ~0.3              -> 0.383 + 0.10 + 0.045 + 0.20 = 0.73 -> 3
#   Hence (0.72, 0.60, 0.48).
THRESHOLDS: dict[str, tuple[float, float, float]] = {
    "tfidf": (0.50, 0.30, 0.10),
    "sbert": (0.62, 0.50, 0.33),
    "bert": (0.72, 0.60, 0.48),
}

# Guardrail: a label of 3 ("strong") must be backed by genuinely strong raw
# semantic similarity, not just feature overlap. Without this, sbert sim 0.30
# (weak for MiniLM) plus maximal feature overlap (0.55) reaches composite
# 0.685 >= 0.62. Pairs failing the floor are capped at 2. TF-IDF keeps a
# floor of 0.0 so its behavior is unchanged.
SIMILARITY_FLOOR_FOR_3: dict[str, float] = {
    "tfidf": 0.0,
    "sbert": 0.45,
    "bert": 0.70,
}


@dataclass(frozen=True)
class PairScore:
    score: int
    confidence: float
    explanation: str


def score_pair(
    co_text: str,
    po_text: str,
    semantic_similarity: float,
    backend: str = "tfidf",
) -> PairScore:
    backend = backend.lower().strip()
    if backend not in THRESHOLDS:
        raise ValueError(f"backend must be one of: {', '.join(sorted(THRESHOLDS))}")

    co_tokens = token_set(co_text)
    po_tokens = token_set(po_text)

    # Bloom levels come from the raw texts (leading-verb policy), not the
    # unordered token sets, so mid-sentence noun-homographs ("the value of
    # assets") cannot inflate the level.
    co_bloom = detect_bloom(co_tokens, text=co_text)
    po_bloom = detect_bloom(po_tokens, text=po_text)
    bloom_gap = bloom_distance(co_bloom, po_bloom)

    # Domains are also detected from the raw texts so multi-word phrases
    # ("code generation", "organizational behavior") actually match.
    co_domains = detect_domains(co_tokens, text=co_text)
    po_domains = detect_domains(po_tokens, text=po_text)
    domain_overlap = jaccard(co_domains, po_domains)
    # A single shared GENERIC domain (everyday tech/business vocabulary:
    # "data", "model", "system", "application", ...) is weak evidence of
    # topical relatedness, so it earns half credit.  Any shared specific
    # domain, or >1 shared domain, earns full credit.
    shared_domains = co_domains & po_domains
    if shared_domains and len(shared_domains) == 1 and shared_domains <= GENERIC_DOMAINS:
        domain_overlap *= 0.5

    token_overlap = jaccard(co_tokens, po_tokens)

    composite = (
        0.45 * semantic_similarity
        + 0.2 * domain_overlap
        + 0.15 * token_overlap
        + 0.2 * max(0.0, 1 - (bloom_gap / 5))
    )

    t3, t2, t1 = THRESHOLDS[backend]
    if composite >= t3:
        label = 3
    elif composite >= t2:
        label = 2
    elif composite >= t1:
        label = 1
    else:
        label = 0

    capped = False
    if label == 3 and semantic_similarity < SIMILARITY_FLOOR_FOR_3[backend]:
        label = 2
        capped = True

    explanation = (
        f"semantic={semantic_similarity:.2f}; domain_overlap={domain_overlap:.2f}; "
        f"token_overlap={token_overlap:.2f}; bloom={co_bloom}->{po_bloom} (gap={bloom_gap})"
    )
    if capped:
        explanation += (
            f"; capped at 2: semantic similarity below the {backend} floor for label 3 "
            f"({SIMILARITY_FLOOR_FOR_3[backend]:.2f})"
        )
    return PairScore(score=label, confidence=round(composite, 3), explanation=explanation)
