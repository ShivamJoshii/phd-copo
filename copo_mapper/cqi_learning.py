"""Learning from the CQI history: did the assumption hold, and what works?

Everything here reads rows from ``store`` (one per CO per offering):

1. ``link_outcomes`` — the **feedback loop**. Each missed CO is paired with
   the same CO's next offering. If attainment rose, the reason/action recorded
   for the miss is *supported*; if not, it is *not supported*. (Improvement is
   evidence for an assumption, not proof: cohorts differ year to year.)
2. ``recommend_actions`` — **k-nearest-neighbour case-based reasoning**: for a
   new miss, find the most similar past misses whose outcome is known and
   rank the actions that were followed by improvement.
3. ``mine_rules`` — **Apriori association rules** of the form
   ``{reason, course type, action} -> improved`` with support / confidence / lift.
4. ``failure_patterns`` — **K-Means** over missed COs (k by silhouette) to
   surface failure patterns, including ones the rule tree has no name for.
5. ``train_next_model`` / ``predict_next`` — a **Random Forest regressor**
   that predicts next offering's attainment for a CO given a planned action,
   reported against the "nothing changes" baseline.
6. ``assumption_scorecard`` / ``action_scorecard`` — the summary numbers.

kNN and Apriori are pure Python. K-Means and the Random Forest need
scikit-learn and degrade to an explanatory message without it.
"""

from __future__ import annotations

import importlib.util
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Sequence

from .cqi_rules import ACTION_CATALOG, REASON_ACTION_CODES
from .features import BLOOM_INDEX

IMPROVE_EPS = 0.05          # scaled-attainment rise that counts as "improved"

# Which input a reason says is the problem (used to check the right thing moved).
REASON_TARGET: dict[str, str] = {
    "pedagogy": "ea",
    "lenient_internal": "ea",
    "assessment_design": "ma",
    "indirect_low": "indirect",
    "overestimation": "direct",
    "prerequisites": "direct",
    "content_gap": "direct",
    "engagement": "direct",
    "borderline": "scaled",
}


def _sem_key(semester: str) -> tuple[int, str]:
    digits = re.findall(r"\d+", semester or "")
    return (int(digits[0]) if digits else 0, (semester or "").lower())


def term_key(row: dict) -> tuple:
    return (str(row.get("academic_year") or ""), _sem_key(str(row.get("semester") or "")))


def final_reason(row: dict) -> str:
    """Faculty-confirmed reason if any, else the rule tree's assumption."""
    return (row.get("faculty_reason") or row.get("rule_reason") or "").strip()


def action_codes(row: dict) -> list[str]:
    return [c.strip() for c in str(row.get("action_code") or "").split(",") if c.strip()]


# ---------------------------------------------------------------------------
# 1. Feedback loop
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FollowUp:
    row: dict                 # the missed offering
    next_row: dict | None     # same course + CO, next offering (None = pending)
    reason: str
    actions: tuple[str, ...]
    delta: float | None       # next scaled - this scaled
    target_delta: float | None  # change in the input the reason points at
    improved: bool | None
    resolved: bool | None     # next offering met its target
    verdict: str              # supported | not supported | pending


def link_outcomes(rows: Sequence[dict]) -> list[FollowUp]:
    """Pair every missed CO with its next offering."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("level") == "CO":
            groups[(row["course_id"], row["outcome_id"])].append(row)
    out: list[FollowUp] = []
    for offerings in groups.values():
        offerings.sort(key=term_key)
        for i, row in enumerate(offerings):
            if row.get("achieved"):
                continue
            nxt = offerings[i + 1] if i + 1 < len(offerings) else None
            reason = final_reason(row)
            if nxt is None:
                out.append(FollowUp(row, None, reason, tuple(action_codes(row)), None, None, None, None, "pending"))
                continue
            delta = float(nxt["scaled"]) - float(row["scaled"])
            target = REASON_TARGET.get(reason, "scaled")
            target_delta = None
            if row.get(target) is not None and nxt.get(target) is not None:
                target_delta = float(nxt[target]) - float(row[target])
            improved = delta > IMPROVE_EPS
            out.append(
                FollowUp(
                    row=row,
                    next_row=nxt,
                    reason=reason,
                    actions=tuple(action_codes(row)),
                    delta=round(delta, 4),
                    target_delta=None if target_delta is None else round(target_delta, 4),
                    improved=improved,
                    resolved=bool(nxt.get("achieved")),
                    verdict="supported" if improved else "not supported",
                )
            )
    return out


def known(followups: Sequence[FollowUp], with_action: bool = True) -> list[FollowUp]:
    return [f for f in followups if f.improved is not None and (f.actions or not with_action)]


def prev_attained(rows: Sequence[dict], course_id: str, outcome_id: str, before: dict) -> float | None:
    """Scaled attainment of the same CO in the latest offering before ``before``."""
    earlier = [
        r for r in rows
        if r.get("level") == "CO" and r["course_id"] == course_id and r["outcome_id"] == outcome_id
        and term_key(r) < term_key(before)
    ]
    if not earlier:
        return None
    return float(max(earlier, key=term_key)["scaled"])


# ---------------------------------------------------------------------------
# 6. Scorecards
# ---------------------------------------------------------------------------


def assumption_scorecard(followups: Sequence[FollowUp]) -> list[dict]:
    """Per reason: how often the next offering improved after acting on it."""
    by_reason: dict[str, list[FollowUp]] = defaultdict(list)
    for f in known(followups):
        by_reason[f.reason].append(f)
    out = []
    for reason, items in by_reason.items():
        n = len(items)
        out.append(
            {
                "reason": reason,
                "cases": n,
                "supported": sum(1 for f in items if f.improved),
                "supported_pct": round(sum(1 for f in items if f.improved) / n * 100, 1),
                "resolved_pct": round(sum(1 for f in items if f.resolved) / n * 100, 1),
                "mean_change": round(statistics.fmean(f.delta for f in items), 3),
            }
        )
    return sorted(out, key=lambda r: (-r["cases"], r["reason"]))


def action_scorecard(followups: Sequence[FollowUp]) -> list[dict]:
    """Per (reason, action): success rate and mean change in attainment."""
    stats: dict[tuple[str, str], list[FollowUp]] = defaultdict(list)
    for f in known(followups):
        for code in f.actions:
            stats[(f.reason, code)].append(f)
    out = []
    for (reason, code), items in stats.items():
        n = len(items)
        out.append(
            {
                "reason": reason,
                "action": code,
                "cases": n,
                "improved_pct": round(sum(1 for f in items if f.improved) / n * 100, 1),
                "mean_change": round(statistics.fmean(f.delta for f in items), 3),
            }
        )
    return sorted(out, key=lambda r: (r["reason"], -r["improved_pct"], -r["cases"]))


# ---------------------------------------------------------------------------
# 2. kNN action recommender (case-based reasoning)
# ---------------------------------------------------------------------------


def _vector(row: dict) -> list[float]:
    bloom = BLOOM_INDEX.get(str(row.get("bloom_level") or ""), None)
    return [
        float(row.get("ma") or 0.0),
        float(row.get("ea") or 0.0),
        float(row.get("indirect") or 0.0),
        0.15 if str(row.get("course_type") or "").lower() == "practical" else 0.0,
        0.0 if bloom is None else 0.15 * bloom / 5,
    ]


@dataclass(frozen=True)
class ActionRecommendation:
    action: str
    label: str
    similar_cases: int
    improved: int
    success_rate: float
    mean_change: float
    examples: tuple[str, ...]


def recommend_actions(
    query: dict, followups: Sequence[FollowUp], k: int = 7
) -> list[ActionRecommendation]:
    """Actions taken in the k most similar past misses, ranked by what followed."""
    cases = known(followups)
    if not cases:
        return []
    q = _vector(query)
    nearest = sorted(cases, key=lambda f: math.dist(q, _vector(f.row)))[: max(1, k)]
    per_action: dict[str, list[FollowUp]] = defaultdict(list)
    for f in nearest:
        for code in f.actions:
            per_action[code].append(f)
    recs = []
    for code, items in per_action.items():
        n = len(items)
        wins = sum(1 for f in items if f.improved)
        recs.append(
            ActionRecommendation(
                action=code,
                label=ACTION_CATALOG[code].label if code in ACTION_CATALOG else code,
                similar_cases=n,
                improved=wins,
                success_rate=round(wins / n, 3),
                mean_change=round(statistics.fmean(f.delta for f in items), 3),
                examples=tuple(
                    f"{f.row['course_id']} {f.row['outcome_id']} {f.row['academic_year']}" for f in items[:3]
                ),
            )
        )
    return sorted(recs, key=lambda r: (-r.success_rate, -r.mean_change, -r.similar_cases))


# ---------------------------------------------------------------------------
# 3. Apriori association rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AssociationRule:
    antecedent: tuple[str, ...]
    consequent: str
    cases: int                # transactions containing the antecedent
    support: float            # share of all transactions with antecedent + consequent
    confidence: float
    lift: float

    def text(self) -> str:
        return "{" + ", ".join(self.antecedent) + "} → " + self.consequent


def _transaction(f: FollowUp) -> frozenset[str]:
    items = {f"reason={f.reason}"} if f.reason else set()
    if f.row.get("course_type"):
        items.add(f"type={f.row['course_type']}")
    if f.row.get("bloom_level"):
        items.add(f"bloom={f.row['bloom_level']}")
    items.update(f"action={code}" for code in f.actions)
    return frozenset(items)


def mine_rules(
    followups: Sequence[FollowUp],
    min_cases: int = 3,
    min_confidence: float = 0.6,
    max_len: int = 3,
) -> list[AssociationRule]:
    """Apriori: frequent antecedents (with an action) -> improved / not improved.

    Level-wise search: an itemset is only extended if it is itself frequent
    (the Apriori pruning property).
    """
    cases = known(followups)
    n = len(cases)
    if n == 0:
        return []
    transactions = [(_transaction(f), bool(f.improved)) for f in cases]
    base_rate = {True: sum(1 for _, y in transactions if y) / n}
    base_rate[False] = 1 - base_rate[True]

    def count(itemset: frozenset[str]) -> tuple[int, int]:
        total = wins = 0
        for items, improved in transactions:
            if itemset <= items:
                total += 1
                wins += improved
        return total, wins

    singles = sorted({item for items, _ in transactions for item in items})
    frequent: dict[frozenset[str], tuple[int, int]] = {}
    level = []
    for item in singles:
        total, wins = count(frozenset([item]))
        if total >= min_cases:
            frequent[frozenset([item])] = (total, wins)
            level.append(frozenset([item]))
    size = 1
    while level and size < max_len:
        size += 1
        candidates = {a | b for a, b in combinations(level, 2) if len(a | b) == size}
        level = []
        for cand in candidates:
            if any(frozenset(sub) not in frequent for sub in combinations(cand, size - 1)):
                continue
            total, wins = count(cand)
            if total >= min_cases:
                frequent[cand] = (total, wins)
                level.append(cand)

    rules: list[AssociationRule] = []
    for itemset, (total, wins) in frequent.items():
        if not any(item.startswith("action=") for item in itemset):
            continue
        for outcome, hits in ((True, wins), (False, total - wins)):
            confidence = hits / total
            if confidence < min_confidence or base_rate[outcome] == 0:
                continue
            rules.append(
                AssociationRule(
                    antecedent=tuple(sorted(itemset)),
                    consequent="improved" if outcome else "not improved",
                    cases=total,
                    support=round(hits / n, 3),
                    confidence=round(confidence, 3),
                    lift=round(confidence / base_rate[outcome], 2),
                )
            )
    return sorted(rules, key=lambda r: (-r.confidence, -r.cases, len(r.antecedent)))


# ---------------------------------------------------------------------------
# 4. K-Means failure patterns
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FailurePattern:
    cluster: int
    size: int
    ma: float
    ea: float
    indirect: float
    top_reason: str
    top_reason_share: float
    note: str


def failure_patterns(rows: Sequence[dict], max_k: int = 6) -> tuple[list[FailurePattern], str]:
    """Cluster missed COs on (Internal, External, Indirect, Internal - External)."""
    missed = [
        r for r in rows
        if r.get("level") == "CO" and not r.get("achieved") and r.get("ma") is not None
    ]
    if len(missed) < 10:
        return [], f"{len(missed)} missed CO record(s) so far; clustering needs at least 10."
    if importlib.util.find_spec("sklearn") is None:
        return [], "scikit-learn is not installed; clustering is unavailable."
    from sklearn.cluster import KMeans  # noqa: PLC0415
    from sklearn.metrics import silhouette_score  # noqa: PLC0415

    X = [[r["ma"], r["ea"], r["indirect"], r["ma"] - r["ea"]] for r in missed]
    best = None
    for k in range(2, min(max_k, len(missed) - 1) + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        if len(set(model.labels_)) < 2:
            continue
        score = silhouette_score(X, model.labels_)
        if best is None or score > best[0]:
            best = (score, k, model)
    if best is None:
        return [], "The missed COs are too similar to split into patterns."
    score, k, model = best
    patterns = []
    for c in range(k):
        members = [r for r, lab in zip(missed, model.labels_) if lab == c]
        reasons = Counter(final_reason(r) or "unlabelled" for r in members)
        top, top_n = reasons.most_common(1)[0]
        share = top_n / len(members)
        centre = model.cluster_centers_[c]
        note = (
            "matches one known reason"
            if share >= 0.6
            else "mixed reasons: possibly a pattern the rule table has no name for"
        )
        patterns.append(
            FailurePattern(c + 1, len(members), round(float(centre[0]), 2), round(float(centre[1]), 2),
                           round(float(centre[2]), 2), top, round(share, 2), note)
        )
    patterns.sort(key=lambda p: -p.size)
    return patterns, f"{len(missed)} missed COs, {k} patterns (silhouette {score:.2f})."


# ---------------------------------------------------------------------------
# 5. Next-offering prediction (Random Forest)
# ---------------------------------------------------------------------------

_ACTION_CODES = sorted(ACTION_CATALOG)
NEXT_FEATURES = ["ma", "ea", "indirect", "scaled", "target", "is_practical", "bloom", *(
    f"action_{c}" for c in _ACTION_CODES
)]
MIN_NEXT_SAMPLES = 20


def _next_features(row: dict, actions: Sequence[str]) -> list[float]:
    bloom = BLOOM_INDEX.get(str(row.get("bloom_level") or ""), -1)
    return [
        float(row.get("ma") or 0.0),
        float(row.get("ea") or 0.0),
        float(row.get("indirect") or 0.0),
        float(row.get("scaled") or 0.0),
        float(row.get("target") or 0.0),
        1.0 if str(row.get("course_type") or "").lower() == "practical" else 0.0,
        float(bloom),
        *(1.0 if c in actions else 0.0 for c in _ACTION_CODES),
    ]


@dataclass
class NextModel:
    n_samples: int
    cv_mae: float | None
    baseline_mae: float | None      # "next = this year" baseline
    importances: list[tuple[str, float]] = field(default_factory=list)
    _reg: object = None


def train_next_model(followups: Sequence[FollowUp]) -> tuple[NextModel | None, str]:
    cases = known(followups, with_action=False)
    if len(cases) < MIN_NEXT_SAMPLES:
        return None, (
            f"{len(cases)} missed CO(s) with a known next offering; the prediction model "
            f"needs {MIN_NEXT_SAMPLES}."
        )
    if importlib.util.find_spec("sklearn") is None:
        return None, "scikit-learn is not installed; the prediction model is unavailable."
    from sklearn.ensemble import RandomForestRegressor  # noqa: PLC0415
    from sklearn.model_selection import KFold, cross_val_predict  # noqa: PLC0415

    X = [_next_features(f.row, f.actions) for f in cases]
    y = [float(f.next_row["scaled"]) for f in cases]

    def new():
        return RandomForestRegressor(n_estimators=200, min_samples_leaf=2, random_state=0)

    predicted = cross_val_predict(new(), X, y, cv=KFold(n_splits=5, shuffle=True, random_state=0))
    cv_mae = statistics.fmean(abs(p - t) for p, t in zip(predicted, y))
    baseline = statistics.fmean(abs(float(f.row["scaled"]) - t) for f, t in zip(cases, y))
    reg = new().fit(X, y)
    importances = sorted(zip(NEXT_FEATURES, (float(v) for v in reg.feature_importances_)), key=lambda t: -t[1])
    model = NextModel(len(cases), round(cv_mae, 3), round(baseline, 3), importances[:8], reg)
    verdict = "beats" if cv_mae < baseline else "does not beat"
    return model, (
        f"Random Forest on {len(cases)} cases: cross-validated error {cv_mae:.3f} (scaled 0-3), "
        f"which {verdict} the 'nothing changes' baseline ({baseline:.3f})."
    )


def predict_next(model: NextModel, row: dict, actions: Sequence[str]) -> float:
    return round(float(model._reg.predict([_next_features(row, actions)])[0]), 2)


def default_actions(reason: str) -> list[str]:
    return list(REASON_ACTION_CODES.get(reason, ("A99",))[:1])


# ---------------------------------------------------------------------------
# Reason & Action Taken report (NBA-style CQI sheet)
# ---------------------------------------------------------------------------


def cqi_report_rows(rows: Sequence[dict], followups: Sequence[FollowUp]) -> list[dict]:
    """One line per non-attained CO/PO, in the layout of an NBA CQI action sheet."""
    from .action_plan import REASON_TAXONOMY  # noqa: PLC0415
    from .cqi_rules import REASON_CODES, SEVERITY_BY_SLAB, action_label  # noqa: PLC0415

    def label(reason_id: str) -> str:
        if not reason_id:
            return ""
        reason = REASON_TAXONOMY.get(reason_id)
        return f"{REASON_CODES.get(reason_id, '')} {reason.label}".strip() if reason else reason_id

    follow = {
        (f.row["academic_year"], f.row["semester"], f.row["course_id"], f.row["outcome_id"]): f
        for f in followups
    }
    out = []
    for r in rows:
        if r.get("achieved"):
            continue
        f = follow.get((r["academic_year"], r["semester"], r["course_id"], r["outcome_id"])) \
            if r.get("level") == "CO" else None
        codes = action_codes(r)
        out.append(
            {
                "Academic year": r["academic_year"],
                "Semester": r["semester"],
                "Course": r["course_id"],
                "Course type": r.get("course_type") or "",
                "CO/PO": r["outcome_id"],
                "Internal": None if r.get("ma") is None else round(r["ma"], 2),
                "External": None if r.get("ea") is None else round(r["ea"], 2),
                "Indirect": None if r.get("indirect") is None else round(r["indirect"], 2),
                "Attained": r.get("scaled"),
                "Target": r.get("target"),
                "Gap": round(float(r.get("gap") or 0.0), 2),
                "Slab": r.get("slab"),
                "Severity": SEVERITY_BY_SLAB.get(r.get("slab"), "").split(":")[0],
                "Weakest input": r.get("weakest") or "",
                "Reason (rule)": label(r.get("rule_reason") or ""),
                "Reason (ML)": label(r.get("ml_reason") or ""),
                "Reason (final)": label(final_reason(r)),
                "Faculty reasoning": r.get("reasoning_text") or "",
                "Action code": ", ".join(action_label(c) for c in codes),
                "Action taken": (r.get("action_plan") or "").replace("\n", " "),
                "Responsible": r.get("responsible") or "",
                "Review date": r.get("review_date") or "",
                "Next attainment": None if f is None or f.next_row is None else f.next_row["scaled"],
                "Improved": "" if f is None or f.improved is None else ("Y" if f.improved else "N"),
            }
        )
    return out
