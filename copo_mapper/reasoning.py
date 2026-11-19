"""Plain-language reasoning for each CO-PO mapping decision.

``scoring.score_pair`` already computes every signal that decides a mapping
strength (semantic similarity, shared domains, shared terms, Bloom levels).
Its ``explanation`` string lists those numbers for developers; this module
turns the same numbers into a short paragraph a reviewer (NBA/accreditation
committee, programme coordinator) can read without knowing the scorer.

Nothing here changes a score. The reasoning only *describes* the decision,
including when a rating rests mostly on Bloom-level alignment rather than
shared subject matter, so the reviewer knows which cells to check by hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .features import BLOOM_INDEX, GENERIC_DOMAINS, stem

STRENGTH_LABELS: dict[int, str] = {
    0: "Not mapped",
    1: "Slight",
    2: "Moderate",
    3: "Substantial",
}

DOMAIN_LABELS: dict[str, str] = {
    "algorithms": "algorithms",
    "databases": "databases",
    "software": "software development",
    "communication": "communication",
    "ethics": "ethics",
    "experiments": "experimentation",
    "compilers": "compilers",
    "systems": "computer systems",
    "networks": "networks",
    "security": "security",
    "ml_data": "data / machine learning",
    "management": "management",
    "finance": "finance",
    "marketing": "marketing",
    "hr_leadership": "HR / leadership",
    "entrepreneurship": "entrepreneurship",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]*")

# Weights of the composite score (must match scoring.score_pair).
W_SEMANTIC, W_DOMAIN, W_TOKEN, W_BLOOM = 0.45, 0.20, 0.15, 0.20


@dataclass(frozen=True)
class ReasoningInputs:
    label: int
    composite: float
    rescaled_similarity: float
    domain_overlap: float
    token_overlap: float
    shared_domains: frozenset[str]
    shared_token_stems: frozenset[str]
    co_text: str
    co_bloom: str
    po_bloom: str
    thresholds: tuple[float, float, float]  # (t3, t2, t1)
    capped: bool = False
    floor_for_3: float = 0.0


def _similarity_phrase(value: float) -> str:
    if value < 0.15:
        return "little to no"
    if value < 0.35:
        return "low"
    if value < 0.55:
        return "moderate"
    return "high"


def shared_terms(co_text: str, shared_stems: frozenset[str], limit: int = 5) -> list[str]:
    """Surface words from the CO text whose stems are shared with the PO."""
    out: list[str] = []
    seen: set[str] = set()
    for word in _WORD_RE.findall(co_text):
        key = stem(word.lower())
        if key in shared_stems and key not in seen:
            seen.add(key)
            out.append(word.lower())
            if len(out) >= limit:
                break
    return out


def _bloom_sentence(co_bloom: str, po_bloom: str) -> str:
    gap = abs(BLOOM_INDEX[co_bloom] - BLOOM_INDEX[po_bloom])
    levels = f"CO is at '{co_bloom.title()}', PO at '{po_bloom.title()}'"
    if gap == 0:
        return f"Cognitive level matches ({levels})."
    if gap == 1:
        return f"Cognitive levels are close ({levels}, 1 level apart)."
    return f"Cognitive levels differ ({levels}, {gap} levels apart)."


def build_pair_reasoning(r: ReasoningInputs) -> str:
    """One short reviewer-facing paragraph explaining the mapping strength."""
    label_text = f"{STRENGTH_LABELS[r.label]} ({r.label})"
    parts: list[str] = [f"{label_text}."]

    # 1. Subject matter
    parts.append(
        f"The CO and PO share {_similarity_phrase(r.rescaled_similarity)} subject matter "
        f"(semantic similarity {r.rescaled_similarity:.2f})."
    )
    if r.shared_domains:
        names = ", ".join(sorted(DOMAIN_LABELS.get(d, d) for d in r.shared_domains))
        if len(r.shared_domains) == 1 and r.shared_domains <= GENERIC_DOMAINS:
            parts.append(f"Both touch on {names} (a broad area, so only weak evidence).")
        else:
            parts.append(f"Both address: {names}.")
    terms = shared_terms(r.co_text, r.shared_token_stems)
    if terms:
        parts.append("Shared key terms: " + ", ".join(f'"{t}"' for t in terms) + ".")
    elif not r.shared_domains:
        parts.append("No key terms or subject areas in common.")

    # 2. Cognitive level
    parts.append(_bloom_sentence(r.co_bloom, r.po_bloom))

    # 3. Where the score came from — flag ratings held up by Bloom alone.
    bloom_part = W_BLOOM * max(0.0, 1 - (abs(BLOOM_INDEX[r.co_bloom] - BLOOM_INDEX[r.po_bloom]) / 5))
    content_part = (
        W_SEMANTIC * r.rescaled_similarity + W_DOMAIN * r.domain_overlap + W_TOKEN * r.token_overlap
    )
    if r.label >= 1 and bloom_part > content_part:
        parts.append(
            "This rating rests mainly on cognitive-level alignment rather than shared content; "
            "please confirm the CO genuinely contributes to this PO."
        )

    # 4. Why not one level higher / lower
    t3, t2, t1 = r.thresholds
    if r.capped:
        parts.append(
            "Overall score qualified for Substantial, but it was held at Moderate because the "
            f"semantic similarity is below the minimum required for a Substantial rating "
            f"({r.floor_for_3:.2f} raw)."
        )
    elif r.label < 3:
        next_cut = {0: t1, 1: t2, 2: t3}[r.label]
        parts.append(
            f"Overall score {r.composite:.2f}; {next_cut:.2f} is needed for "
            f"{STRENGTH_LABELS[r.label + 1]}."
        )
    else:
        parts.append(f"Overall score {r.composite:.2f} (≥ {t3:.2f} for Substantial).")

    return " ".join(parts)
