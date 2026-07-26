from dataclasses import dataclass

from .features import (
    GENERIC_DOMAINS,
    bloom_distance,
    detect_bloom,
    detect_domains,
    jaccard,
    token_set,
)

# ---------------------------------------------------------------------------
# Per-backend similarity normalization (the calibration fix)
# ---------------------------------------------------------------------------
#
# The composite is 0.45 * semantic + up to 0.55 from features (0.2 domain
# overlap + 0.15 token overlap + 0.2 bloom-proximity). Measured on the real
# MBA exports (1,980 CO x PO pairs): the domain term is 0 for 93% of pairs
# (mean 0.007) and the bloom term averages ~0.13 (worst-case default 0.20
# when neither text carries an action verb). So for typical pairs the
# composite is dominated by the semantic term plus ~0.13.
#
# Raw cosine similarity, however, lives on a *different scale per backend*:
#
#   tfidf                  : ~0 for unrelated texts, grows toward 1 only with
#                            heavy term overlap. Already effectively [0, 1].
#   sbert (all-MiniLM-L6-v2): unrelated sentence pairs sit ~0.0-0.25, weak
#                            ~0.25-0.45, moderate ~0.45-0.6, strong/near-
#                            paraphrase > 0.6-0.75.
#   bert (bert-base-uncased, mean-pooled): anisotropic embeddings push every
#                            cosine high — unrelated ~0.5-0.6, related
#                            ~0.7-0.9.
#
# Feeding these raw cosines into one composite formula therefore requires
# either per-backend composite thresholds (the previous design) or
# normalizing the similarity first. The previous per-backend thresholds were
# derived analytically ASSUMING the old feature behavior (bloom ~0.20 floor,
# generous domain credit); after the feature pruning the real feature
# contribution dropped to ~0.13-0.15, so a genuinely strong sbert pair
# (sim ~0.65, little lexical overlap) reached only ~0.44 composite — below
# even the old sbert t2 of 0.50 — and nearly every predicted strength
# collapsed to 0.
#
# The fix: rescale each backend's raw cosine onto a common [0, 1] scale
# BEFORE the composite, so every backend feeds the same scale and shares the
# single threshold set that IS validated on real data (the tfidf cutoffs).
#
# (lo, hi) anchors per backend: rescaled = clamp((raw - lo) / (hi - lo), 0, 1).
#   lo = "unrelated floor"  (raw cosine of clearly unrelated outcome pairs)
#   hi = "near-paraphrase ceiling" (raw cosine of near-identical statements)
#
#   tfidf: (0.0, 1.0) — identity. TF-IDF cosines already live on [0, ~1]
#     with 0 meaning unrelated; the identity mapping keeps the tfidf path
#     byte-for-byte identical to its pre-refactor (empirically validated)
#     behavior.
#   sbert: (0.25, 0.75) — published/well-known MiniLM cosine distribution:
#     ~0.25 is the unrelated floor, ~0.75 a near-paraphrase ceiling for this
#     model family. (sentence-transformers could not be installed in the
#     calibration sandbox, so these rest on the published distribution;
#     scripts/calibrate_sbert.py re-derives them empirically when run.)
#   bert: (0.55, 0.90) — mean-pooled bert-base anisotropy: unrelated pairs
#     rarely fall below ~0.55, and ~0.90 is a near-paraphrase ceiling.
SIMILARITY_RESCALE: dict[str, tuple[float, float]] = {
    "tfidf": (0.0, 1.0),
    "sbert": (0.25, 0.75),
    "bert": (0.55, 0.90),
}


def rescale_similarity(similarity: float, backend: str) -> float:
    """Map a backend's raw cosine onto the common [0, 1] semantic scale.

    Affine rescale with clamping using the SIMILARITY_RESCALE anchors.
    The (0.0, 1.0) anchors (tfidf) are treated as a true identity — no
    clamping — preserving the legacy tfidf behavior byte-for-byte.
    """
    lo, hi = SIMILARITY_RESCALE[backend]
    if lo == 0.0 and hi == 1.0:
        return similarity
    return min(1.0, max(0.0, (similarity - lo) / (hi - lo)))


# Composite-score cutoffs (t3, t2, t1): label 3 if composite >= t3, 2 if
# >= t2, 1 if >= t1, else 0.
#
# Because the semantic term is normalized per backend (SIMILARITY_RESCALE),
# every backend now shares the SAME cutoffs — the (0.50, 0.30, 0.10) set that
# is empirically validated on the real institutional exports via the tfidf
# pipeline (program-wide label distribution 0:26% / 1:72% / 2:1.7% / 3:0.2%,
# with every label-3 pair a genuinely strong match). The dict form is kept
# so a backend could be given override cutoffs later, but all entries are
# intentionally identical today.
#
# Band arithmetic on the common scale (sbert anchors 0.25/0.75; bloom term
# shown at its measured typical value ~0.13 unless noted):
#   unrelated: raw 0.20 -> rescaled 0.00 -> composite = features only
#              (~0.13 typical; 0.20 at the verb-less bloom-default worst
#              case) -> label 1, never 2 (0.20 < t2 = 0.30); with any real
#              bloom gap >= 3 the composite drops below 0.10 -> label 0.
#   weak:      raw 0.35 -> rescaled 0.20 -> 0.09 + ~0.13       = 0.22 -> 1.
#   moderate:  raw 0.50 -> rescaled 0.50 -> 0.225 + ~0.15      = 0.38 -> 2.
#   strong:    raw 0.65 -> rescaled 0.80 -> 0.36 + ~0.18       = 0.54 -> 3
#              (raw 0.65 also clears the 0.45 label-3 floor below).
#
# Same check for bert (anchors 0.55/0.90):
#   unrelated: raw 0.55 -> 0.00  -> <= 0.20        -> 0 or low 1.
#   weak:      raw 0.65 -> 0.286 -> 0.129 + ~0.13  = 0.26 -> 1.
#   moderate:  raw 0.75 -> 0.571 -> 0.257 + ~0.15  = 0.41 -> 2.
#   strong:    raw 0.85 -> 0.857 -> 0.386 + ~0.18  = 0.57 -> 3 (floor 0.70 ok).
THRESHOLDS: dict[str, tuple[float, float, float]] = {
    "tfidf": (0.50, 0.30, 0.10),
    "sbert": (0.50, 0.30, 0.10),
    "bert": (0.50, 0.30, 0.10),
}

# Guardrail: a label of 3 ("strong") must be backed by genuinely strong RAW
# semantic similarity, not just feature overlap. Without this, sbert raw sim
# 0.30 (weak for MiniLM; rescaled 0.10) plus maximal feature overlap (0.55)
# reaches composite 0.595 >= 0.50. The floor is applied to the RAW cosine —
# 0.45 raw sits at the MiniLM moderate/strong boundary, 0.70 raw at the
# mean-pooled-BERT related boundary. Pairs failing the floor are capped at 2.
# TF-IDF keeps a floor of 0.0 so its behavior is unchanged.
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

    # Normalize the raw cosine onto the common [0, 1] scale so all backends
    # share one validated threshold set (see SIMILARITY_RESCALE).
    rescaled_similarity = rescale_similarity(semantic_similarity, backend)

    composite = (
        0.45 * rescaled_similarity
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

    # The label-3 floor is checked against the RAW similarity: strong claims
    # need strong raw semantics regardless of feature overlap.
    capped = False
    if label == 3 and semantic_similarity < SIMILARITY_FLOOR_FOR_3[backend]:
        label = 2
        capped = True

    lo, hi = SIMILARITY_RESCALE[backend]
    if lo == 0.0 and hi == 1.0:
        # Identity mapping (tfidf): keep the legacy explanation format so
        # the tfidf path stays byte-for-byte identical.
        semantic_part = f"semantic={semantic_similarity:.2f}"
    else:
        # Show both the normalized value (what the composite uses) and the
        # raw cosine (what the model produced) so nothing is hidden.
        semantic_part = (
            f"semantic={rescaled_similarity:.2f} (raw {semantic_similarity:.2f})"
        )
    explanation = (
        f"{semantic_part}; domain_overlap={domain_overlap:.2f}; "
        f"token_overlap={token_overlap:.2f}; bloom={co_bloom}->{po_bloom} (gap={bloom_gap})"
    )
    if capped:
        explanation += (
            f"; capped at 2: raw semantic similarity below the {backend} floor for label 3 "
            f"({SIMILARITY_FLOOR_FOR_3[backend]:.2f})"
        )
    return PairScore(score=label, confidence=round(composite, 3), explanation=explanation)
