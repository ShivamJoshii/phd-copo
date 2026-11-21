"""Synthetic multi-semester CQI history, for demonstrating the learning loop.

Real institutions need 2-3 years of records before the feedback loop, the kNN
recommender, the Apriori rules and the prediction model have anything to learn
from. This module generates clearly-labelled demo offerings (course ids start
with ``DEMO-`` and ``is_demo = 1``) so every screen can be shown working.

**These numbers are simulated, not evidence.** The simulation assumes each
weak CO has a hidden true cause; an action that matches the cause usually
lifts the affected input next year, a mismatched action usually does not.
Whatever the learning layer "discovers" on demo data is this assumption being
recovered, which tests the method, not the pedagogy.

All attainment values go through the production pipeline
(compute_co_attainment -> diagnose_co -> suggest_reason_co).
"""

from __future__ import annotations

import random

from .action_plan import suggest_reason_co
from .attainment import COAttainmentInput, WeightConfig, compute_co_attainment
from .cqi_rules import ACTION_CATALOG, COURSE_TYPES, REASON_ACTION_CODES, slab
from .diagnostics import diagnose_co
from .features import BLOOM_ORDER

DEMO_PREFIX = "DEMO-"
DEMO_YEARS = ("2023-24", "2024-25", "2025-26")
DEMO_TARGET = 2.0

# Hidden true cause -> starting (Internal, External, Indirect) pattern.
_CAUSE_PATTERNS: dict[str, tuple[float, float, float]] = {
    "healthy": (0.78, 0.74, 0.80),
    "prerequisites": (0.44, 0.47, 0.64),
    "content_gap": (0.60, 0.61, 0.78),
    "assessment_design": (0.50, 0.76, 0.74),
    "pedagogy": (0.74, 0.60, 0.72),
    "lenient_internal": (0.82, 0.50, 0.74),
    "indirect_low": (0.70, 0.68, 0.30),
    "overestimation": (0.45, 0.46, 0.84),
}
_CAUSE_WEIGHTS = [5, 2, 2, 2, 2, 2, 1, 1]

# Which inputs rise when the right action is taken for a cause.
_CAUSE_LIFTS: dict[str, tuple[str, ...]] = {
    "prerequisites": ("ma", "ea"),
    "content_gap": ("ma", "ea"),
    "assessment_design": ("ma",),
    "pedagogy": ("ea",),
    "lenient_internal": ("ea",),
    "indirect_low": ("indirect",),
    "overestimation": ("ma", "ea"),
}

_COURSES = ["CS301", "CS302", "CS305", "CS401", "CS404", "EC301", "EC305", "ME302", "ME306",
            "MB101", "MB204", "CS351L", "CS352L", "EC351L", "ME351L", "CS451L"]


def _clip(x: float) -> float:
    return min(0.98, max(0.05, x))


def generate_demo_rows(n_courses: int = 16, cos_per_course: int = 5, seed: int = 0) -> list[dict]:
    """Store rows for ``n_courses`` demo courses over three academic years."""
    rng = random.Random(seed)
    config = WeightConfig(0.4, 0.6, 0.8, 0.2, DEMO_TARGET, DEMO_TARGET)
    all_codes = [c for c in ACTION_CATALOG if c not in ("A12", "A13", "A99")]
    rows: list[dict] = []

    for name in _COURSES[:n_courses]:
        course_type = COURSE_TYPES[1] if name.endswith("L") else COURSE_TYPES[0]
        class_size = rng.choice([30, 45, 60, 60, 72, 120])
        semester = rng.choice(["3", "4", "5", "6", "7"])
        for c in range(1, cos_per_course + 1):
            cause = rng.choices(list(_CAUSE_PATTERNS), weights=_CAUSE_WEIGHTS)[0]
            state = {
                key: _clip(base + rng.uniform(-0.05, 0.05))
                for key, base in zip(("ma", "ea", "indirect"), _CAUSE_PATTERNS[cause])
            }
            bloom = rng.choice(BLOOM_ORDER[1:])
            map_strength = round(rng.uniform(1.0, 3.0), 2)
            for year in DEMO_YEARS:
                (res,) = compute_co_attainment(
                    [COAttainmentInput(f"CO{c}", state["ma"], state["ea"], state["indirect"])], config
                )
                exp = diagnose_co(res, config)
                suggestion = suggest_reason_co(exp)
                row = {
                    "academic_year": year,
                    "semester": semester,
                    "course_id": DEMO_PREFIX + name,
                    "level": "CO",
                    "outcome_id": f"CO{c}",
                    "course_type": course_type,
                    "class_size": class_size,
                    "ma": round(state["ma"], 4),
                    "ea": round(state["ea"], 4),
                    "indirect": round(state["indirect"], 4),
                    "direct": res.direct_attainment,
                    "final": res.final_attainment,
                    "scaled": res.scaled_attainment,
                    "target": DEMO_TARGET,
                    "gap": max(0.0, exp.gap),
                    "achieved": 1 if exp.achieved else 0,
                    "slab": slab(res.scaled_attainment),
                    "bloom_level": bloom,
                    "map_strength": map_strength,
                    "weakest": exp.weakest_component,
                    "rule_reason": suggestion.reason_id if suggestion else "",
                    "ml_reason": "",
                    "faculty_reason": "",
                    "reasoning_text": "",
                    "action_code": "",
                    "action_plan": "",
                    "responsible": "",
                    "review_date": "",
                    "is_demo": 1,
                }
                lift_keys: tuple[str, ...] = ()
                if suggestion is not None:
                    # Faculty mostly accept the assumption; sometimes they know the real cause.
                    knows_cause = cause != "healthy" and rng.random() < 0.35
                    reason = cause if knows_cause else suggestion.reason_id
                    # Most take an action recommended for their reason; some pick something else.
                    if rng.random() < 0.75:
                        code = rng.choice(REASON_ACTION_CODES[reason])
                    else:
                        code = rng.choice(all_codes)
                    row.update(
                        faculty_reason=reason,
                        action_code=code,
                        action_plan=ACTION_CATALOG[code].text(course_type),
                        reasoning_text="demo record (simulated)",
                        responsible="Demo Faculty",
                    )
                    matches = cause != "healthy" and code in REASON_ACTION_CODES.get(cause, ())
                    if matches and rng.random() < 0.8:
                        lift_keys = _CAUSE_LIFTS.get(cause, ())
                rows.append(row)
                # Evolve to next year: targeted lift if the action matched, noise otherwise.
                for key in state:
                    change = rng.uniform(-0.02, 0.02)
                    if key in lift_keys:
                        change += rng.uniform(0.07, 0.15)
                    state[key] = _clip(state[key] + change)
    return rows
