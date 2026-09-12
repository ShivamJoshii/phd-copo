"""Decision-tree reasoning and CQI action plans for missed CO/PO targets.

This is the "close the loop" layer of the CQI cycle. ``diagnostics`` answers
*where* a miss comes from arithmetically (weakest input, cheapest lever);
this module answers *why* in pedagogical terms and *what to do about it*:

1. **Expert decision tree** (``suggest_reason_co`` / ``suggest_reason_po``) —
   a deterministic, fully auditable tree over the attainment components that
   lands on a *reason category* from ``REASON_TAXONOMY`` with a recorded
   decision path and a CQI action-plan template. Works from the first course
   analysed, no training data needed.
2. **Faculty reasoning capture** (``ActionPlanRecord``) — the faculty
   confirms or overrides the suggested reason, adds free-text reasoning and
   the corrective action they commit to. Records round-trip through CSV so
   they accumulate across courses/semesters (Streamlit Cloud has no durable
   disk).
3. **Learned decision tree** (``train_reason_tree`` / ``predict_reason``) —
   an interpretable scikit-learn ``DecisionTreeClassifier`` trained on the
   accumulated faculty-labelled records. Once enough labelled cases exist it
   predicts the likely reason for a *new* miss, and its learned rules are
   exported as text so every prediction is inspectable. Guarded import, same
   pattern as the SBERT/BERT backends: without scikit-learn everything else
   still works and callers get an explanatory message instead of a model.

Small-data honesty: the learned tree refuses to train below
``MIN_TRAIN_SAMPLES`` labelled CO records or with fewer than two distinct
reason labels, and reports cross-validated accuracy only when every class
has enough members for a stratified split.
"""

from __future__ import annotations

import csv
import importlib.util
import statistics
from dataclasses import asdict, dataclass, field
from io import StringIO
from typing import Sequence

from .diagnostics import COExplanation, CourseDiagnosis, POExplanation

# ---------------------------------------------------------------------------
# Reason taxonomy (standard OBE/CQI categories)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reason:
    reason_id: str
    label: str
    description: str
    actions: tuple[str, ...]


REASON_TAXONOMY: dict[str, Reason] = {
    reason.reason_id: reason
    for reason in (
        Reason(
            "prerequisites",
            "Weak student prerequisites",
            "Students lack the foundation the outcome builds on; both internal and "
            "external assessment collapse together and the gap is large.",
            (
                "Run a diagnostic/prerequisite test in week 1 and publish the gap profile.",
                "Schedule remedial / bridge classes for the weak prerequisite topics.",
                "Pair weak students with mentors; track their internal scores separately.",
            ),
        ),
        Reason(
            "content_gap",
            "Content / syllabus coverage gap",
            "Both assessment arms are moderately below target: the material for this "
            "outcome was not covered deeply (or recently) enough.",
            (
                "Revise the lesson plan to give this CO's topics more contact hours.",
                "Add tutorial sheets / content beyond syllabus for the weak topics.",
                "Re-check that question papers actually sample every unit mapped to the CO.",
            ),
        ),
        Reason(
            "assessment_design",
            "Internal assessment design",
            "Internal (MA) is well below target while the external (EA) arm is fine — "
            "the internal instrument, not the learning, is the outlier.",
            (
                "Re-map internal question papers to COs and Bloom levels; fix over-hard items.",
                "Moderate internal marking with a rubric shared across sections.",
                "Rebalance the internal components (quiz/assignment/mid-term) feeding this CO.",
            ),
        ),
        Reason(
            "pedagogy",
            "Teaching–learning method",
            "External (EA) lags while internal (MA) is fine: students perform in "
            "class-proximate assessment but not under end-semester conditions.",
            (
                "Introduce active-learning treatment for the CO's topics (case study / lab / flipped).",
                "Add exam-style practice: previous-year and problem-solving sessions.",
                "Collect topic-wise EA item analysis and reteach the worst two topics.",
            ),
        ),
        Reason(
            "engagement",
            "Student engagement / attendance",
            "All three inputs are far below target together — a systemic engagement "
            "problem rather than a single instrument or topic.",
            (
                "Review attendance/participation data for the cohort; counsel outliers.",
                "Add continuous low-stakes assessment to force regular practice.",
                "Involve mentors/parents per institutional early-warning policy.",
            ),
        ),
        Reason(
            "indirect_low",
            "Indirect attainment (survey) low",
            "The indirect arm (course-exit survey / feedback) is the weakest input; "
            "direct evidence is comparatively healthy.",
            (
                "Check survey administration: timing, coverage, and student awareness of COs.",
                "Communicate 'closing the loop': show students how past feedback changed the course.",
                "Review survey instrument wording against the CO statements.",
            ),
        ),
        Reason(
            "curriculum_alignment",
            "Curriculum / mapping alignment",
            "PO-level: the contributing COs are healthy (or absent), yet the PO misses — "
            "the articulation between courses and this PO is the weak link.",
            (
                "Review the CO–PO mapping strengths feeding this PO with the programme committee.",
                "Add or strengthen course content that genuinely addresses this PO.",
                "Ensure at least one strongly-mapped (level 3) CO exists for this PO in the curriculum.",
            ),
        ),
        Reason(
            "other",
            "Other (custom)",
            "Faculty-identified cause outside the standard taxonomy.",
            ("Document the cause and the corrective action taken.",),
        ),
    )
}

REASON_ORDER: list[str] = list(REASON_TAXONOMY)

# Severity cutoffs used by the expert tree (fractions of the required level).
SEVERE_GAP_RATIO = 0.25
SYSTEMIC_GAP_RATIO = 0.33


@dataclass(frozen=True)
class ReasonSuggestion:
    reason_id: str
    label: str
    decision_path: tuple[str, ...]   # human-readable branch taken at each node
    actions: tuple[str, ...]


def _suggestion(reason_id: str, path: list[str]) -> ReasonSuggestion:
    reason = REASON_TAXONOMY[reason_id]
    return ReasonSuggestion(
        reason_id=reason_id,
        label=reason.label,
        decision_path=tuple(path),
        actions=reason.actions,
    )


# ---------------------------------------------------------------------------
# Expert decision tree
# ---------------------------------------------------------------------------


def suggest_reason_co(exp: COExplanation) -> ReasonSuggestion | None:
    """Walk the expert decision tree for one missed CO.

    Returns None when the CO met its target. Every branch taken is recorded
    in ``decision_path`` so the UI (and the thesis) can show the exact route
    to the conclusion.
    """
    if exp.achieved:
        return None

    needed = exp.target / 3.0  # required level on the component (0..1) scale
    ma = exp.components.get("MA", 0.0)
    ea = exp.components.get("EA", 0.0)
    indirect = exp.components.get("Indirect", 0.0)
    gap_ratio = (exp.gap / exp.target) if exp.target else 0.0

    path = [
        f"{exp.co_id} missed: scaled {exp.scaled:.2f} < target {exp.target:.2f} "
        f"(gap ratio {gap_ratio:.0%}); required component level ≈ {needed:.2f}",
    ]

    all_low = ma < needed and ea < needed and indirect < needed
    if all_low and gap_ratio > SYSTEMIC_GAP_RATIO:
        path.append(
            f"MA {ma:.2f}, EA {ea:.2f} and Indirect {indirect:.2f} are ALL below "
            f"{needed:.2f} with a large gap (> {SYSTEMIC_GAP_RATIO:.0%}) → systemic engagement problem"
        )
        return _suggestion("engagement", path)
    path.append(
        "Not a systemic collapse (some input near target or gap moderate) → inspect individual inputs"
    )

    if exp.weakest_component == "Indirect" and indirect < needed:
        path.append(
            f"Indirect ({indirect:.2f}) is the weakest input and below {needed:.2f}, "
            f"while direct evidence is comparatively healthier → indirect/survey issue"
        )
        return _suggestion("indirect_low", path)

    if ma < needed and ea < needed:
        path.append(f"Both MA ({ma:.2f}) and EA ({ea:.2f}) are below {needed:.2f}")
        if gap_ratio > SEVERE_GAP_RATIO:
            path.append(
                f"Gap ratio {gap_ratio:.0%} > {SEVERE_GAP_RATIO:.0%} → severe joint deficit "
                "points at missing prerequisites"
            )
            return _suggestion("prerequisites", path)
        path.append(
            f"Gap ratio {gap_ratio:.0%} ≤ {SEVERE_GAP_RATIO:.0%} → moderate joint deficit "
            "points at content coverage"
        )
        return _suggestion("content_gap", path)

    if ma < needed <= ea:
        path.append(
            f"MA ({ma:.2f}) below {needed:.2f} while EA ({ea:.2f}) meets it → the internal "
            "instrument is the outlier"
        )
        return _suggestion("assessment_design", path)

    if ea < needed <= ma:
        path.append(
            f"EA ({ea:.2f}) below {needed:.2f} while MA ({ma:.2f}) meets it → end-semester "
            "performance lags classroom-proximate assessment"
        )
        return _suggestion("pedagogy", path)

    # Weighted combination missed even though no single component is below the
    # naive requirement (possible with skewed weights).
    path.append(
        "No single component is below the required level; the weighted combination "
        "still misses → treat as content depth issue and review the weight split"
    )
    return _suggestion("content_gap", path)


def suggest_reason_po(
    exp: POExplanation,
    co_suggestions: dict[str, ReasonSuggestion],
) -> ReasonSuggestion | None:
    """Expert tree for one missed PO, tracing through its dragging COs.

    ``co_suggestions`` maps co_id -> the CO-level suggestion (only missed COs
    present). A missed PO inherits the modal reason of the missed COs that
    drag it, weighted by drag; with no dragging misses it is an alignment
    problem by construction.
    """
    if exp.achieved:
        return None

    path = [
        f"{exp.po_id} missed: scaled {exp.scaled:.2f} < target {exp.target:.2f} (gap {exp.gap:.2f})",
    ]

    if not exp.contributions:
        path.append("No CO maps to this PO at all → curriculum articulation gap")
        return _suggestion("curriculum_alignment", path)

    draggers = [c for c in exp.contributions if c.drag_scaled > 0 and c.co_id in co_suggestions]
    if not draggers:
        path.append(
            "Contributing COs met their targets (or none of the dragging COs missed), "
            "yet the PO is below target → mapping strengths sit on the weaker COs; "
            "curriculum/mapping alignment issue"
        )
        return _suggestion("curriculum_alignment", path)

    # Weight each dragging CO's reason by its drag and take the heaviest.
    drag_by_reason: dict[str, float] = {}
    for c in draggers:
        rid = co_suggestions[c.co_id].reason_id
        drag_by_reason[rid] = drag_by_reason.get(rid, 0.0) + c.drag_scaled
    top_reason = max(drag_by_reason, key=drag_by_reason.get)
    dragged_ids = ", ".join(f"{c.co_id} (drag {c.drag_scaled:.2f})" for c in draggers[:4])
    path.append(f"Dragged below target by missed CO(s): {dragged_ids}")
    path.append(
        f"Dominant underlying CO-level reason (drag-weighted): "
        f"{REASON_TAXONOMY[top_reason].label} → fix the PO through those COs"
    )
    return _suggestion(top_reason, path)


def suggest_for_course(diagnosis: CourseDiagnosis) -> dict[tuple[str, str], ReasonSuggestion]:
    """Suggestions for every missed outcome: {("CO"|"PO", outcome_id): suggestion}."""
    out: dict[tuple[str, str], ReasonSuggestion] = {}
    co_map: dict[str, ReasonSuggestion] = {}
    for exp in diagnosis.missed_cos:
        suggestion = suggest_reason_co(exp)
        if suggestion is not None:
            co_map[exp.co_id] = suggestion
            out[("CO", exp.co_id)] = suggestion
    for exp in diagnosis.missed_pos:
        suggestion = suggest_reason_po(exp, co_map)
        if suggestion is not None:
            out[("PO", exp.po_id)] = suggestion
    return out


# ---------------------------------------------------------------------------
# Faculty action-plan records (the labelled dataset)
# ---------------------------------------------------------------------------

RECORD_FIELDS = [
    "course_id",
    "level",            # "CO" or "PO"
    "outcome_id",
    "ma",
    "ea",
    "indirect",
    "final",
    "target",
    "gap",
    "weakest",
    "suggested_reason",
    "reason",           # faculty-confirmed reason_id (the ML label)
    "reasoning_text",
    "action_plan",
]


@dataclass
class ActionPlanRecord:
    course_id: str
    level: str
    outcome_id: str
    ma: float | None
    ea: float | None
    indirect: float | None
    final: float
    target: float
    gap: float
    weakest: str
    suggested_reason: str
    reason: str
    reasoning_text: str = ""
    action_plan: str = ""

    def to_row(self) -> dict[str, str]:
        row = asdict(self)
        for key in ("ma", "ea", "indirect"):
            row[key] = "" if row[key] is None else f"{row[key]:.4f}"
        for key in ("final", "target", "gap"):
            row[key] = f"{float(row[key]):.4f}"
        return {k: str(row[k]) for k in RECORD_FIELDS}


def record_from_co(
    exp: COExplanation,
    *,
    course_id: str,
    suggested_reason: str,
    reason: str,
    reasoning_text: str = "",
    action_plan: str = "",
) -> ActionPlanRecord:
    return ActionPlanRecord(
        course_id=course_id,
        level="CO",
        outcome_id=exp.co_id,
        ma=exp.components.get("MA"),
        ea=exp.components.get("EA"),
        indirect=exp.components.get("Indirect"),
        final=exp.scaled / 3.0,
        target=exp.target,
        gap=exp.gap,
        weakest=exp.weakest_component,
        suggested_reason=suggested_reason,
        reason=reason,
        reasoning_text=reasoning_text,
        action_plan=action_plan,
    )


def record_from_po(
    exp: POExplanation,
    *,
    course_id: str,
    suggested_reason: str,
    reason: str,
    reasoning_text: str = "",
    action_plan: str = "",
) -> ActionPlanRecord:
    worst = exp.contributions[0].co_id if exp.contributions else ""
    return ActionPlanRecord(
        course_id=course_id,
        level="PO",
        outcome_id=exp.po_id,
        ma=None,
        ea=None,
        indirect=None,
        final=exp.scaled / 3.0,
        target=exp.target,
        gap=exp.gap,
        weakest=worst,
        suggested_reason=suggested_reason,
        reason=reason,
        reasoning_text=reasoning_text,
        action_plan=action_plan,
    )


def records_to_csv(records: Sequence[ActionPlanRecord]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=RECORD_FIELDS)
    writer.writeheader()
    for record in records:
        writer.writerow(record.to_row())
    return buffer.getvalue()


def records_from_csv(text: str) -> list[ActionPlanRecord]:
    """Parse records back; malformed rows are skipped rather than fatal."""
    records: list[ActionPlanRecord] = []
    for row in csv.DictReader(StringIO(text)):
        level = str(row.get("level") or "").strip().upper()
        outcome_id = str(row.get("outcome_id") or "").strip()
        if level not in ("CO", "PO") or not outcome_id:
            continue
        try:
            records.append(
                ActionPlanRecord(
                    course_id=str(row.get("course_id", "")).strip(),
                    level=level,
                    outcome_id=outcome_id,
                    ma=_opt_float(row.get("ma")),
                    ea=_opt_float(row.get("ea")),
                    indirect=_opt_float(row.get("indirect")),
                    final=float(row.get("final", 0.0) or 0.0),
                    target=float(row.get("target", 0.0) or 0.0),
                    gap=float(row.get("gap", 0.0) or 0.0),
                    weakest=str(row.get("weakest", "")).strip(),
                    suggested_reason=str(row.get("suggested_reason", "")).strip(),
                    reason=str(row.get("reason", "")).strip(),
                    reasoning_text=str(row.get("reasoning_text", "")),
                    action_plan=str(row.get("action_plan", "")),
                )
            )
        except (TypeError, ValueError):
            continue
    return records


def _opt_float(value: object) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def merge_records(
    existing: Sequence[ActionPlanRecord], incoming: Sequence[ActionPlanRecord]
) -> list[ActionPlanRecord]:
    """Merge by (course, level, outcome); incoming rows win."""
    merged = {(r.course_id, r.level, r.outcome_id): r for r in existing}
    for record in incoming:
        merged[(record.course_id, record.level, record.outcome_id)] = record
    return list(merged.values())


# ---------------------------------------------------------------------------
# Learned decision tree (scikit-learn, guarded)
# ---------------------------------------------------------------------------

MIN_TRAIN_SAMPLES = 12
FEATURE_NAMES = ["ma", "ea", "indirect", "final", "gap", "weak_MA", "weak_EA", "weak_Indirect"]


def _featurize(record: ActionPlanRecord) -> list[float] | None:
    if record.level != "CO" or not record.reason:
        return None
    if record.ma is None or record.ea is None or record.indirect is None:
        return None
    return [
        record.ma,
        record.ea,
        record.indirect,
        record.final,
        record.gap,
        1.0 if record.weakest == "MA" else 0.0,
        1.0 if record.weakest == "EA" else 0.0,
        1.0 if record.weakest == "Indirect" else 0.0,
    ]


@dataclass
class TrainedReasonTree:
    n_samples: int
    classes: list[str]
    class_counts: dict[str, int]
    rules_text: str
    cv_accuracy: float | None
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    _clf: object = None


def train_reason_tree(
    records: Sequence[ActionPlanRecord],
    min_samples: int = MIN_TRAIN_SAMPLES,
    max_depth: int = 3,
) -> tuple[TrainedReasonTree | None, str]:
    """Train the interpretable reason classifier on faculty-labelled records.

    Returns (model, message). model is None whenever training is not yet
    honest (too few samples / classes) or scikit-learn is unavailable; the
    message always says why or reports quality.
    """
    rows: list[list[float]] = []
    labels: list[str] = []
    for record in records:
        features = _featurize(record)
        if features is not None:
            rows.append(features)
            labels.append(record.reason)

    if len(rows) < min_samples:
        return None, (
            f"{len(rows)} labelled CO record(s) so far — the learned tree unlocks at "
            f"{min_samples}. Until then the expert decision tree provides suggestions."
        )
    distinct = sorted(set(labels))
    if len(distinct) < 2:
        return None, (
            f"All {len(rows)} labelled records share one reason ('{distinct[0]}'); "
            "a classifier needs at least two distinct reasons."
        )
    if importlib.util.find_spec("sklearn") is None:
        return None, (
            "scikit-learn is not installed in this runtime; expert-tree suggestions "
            "remain available. Add scikit-learn to requirements to enable the learned tree."
        )

    from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: PLC0415
    from sklearn.tree import DecisionTreeClassifier, export_text  # noqa: PLC0415

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=0,
    )
    clf.fit(rows, labels)

    counts = {label: labels.count(label) for label in distinct}
    min_class = min(counts.values())
    cv_accuracy: float | None = None
    if min_class >= 2 and len(rows) >= min_samples:
        folds = min(3, min_class)
        try:
            scores = cross_val_score(
                DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=0,
                ),
                rows,
                labels,
                cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=0),
            )
            cv_accuracy = round(float(statistics.fmean(scores)), 3)
        except ValueError:
            cv_accuracy = None

    model = TrainedReasonTree(
        n_samples=len(rows),
        classes=distinct,
        class_counts=counts,
        rules_text=export_text(clf, feature_names=FEATURE_NAMES),
        cv_accuracy=cv_accuracy,
        _clf=clf,
    )
    quality = (
        f"trained on {len(rows)} records, {len(distinct)} reasons"
        + (f", {cv_accuracy:.0%} cross-validated accuracy" if cv_accuracy is not None else
           ", too few per-class samples for cross-validation")
    )
    return model, quality


def predict_reason(
    model: TrainedReasonTree, exp: COExplanation
) -> tuple[str, float] | None:
    """Predict (reason_id, probability) for a missed CO from the learned tree."""
    if model._clf is None or exp.achieved:
        return None
    features = [
        exp.components.get("MA", 0.0),
        exp.components.get("EA", 0.0),
        exp.components.get("Indirect", 0.0),
        exp.scaled / 3.0,
        exp.gap,
        1.0 if exp.weakest_component == "MA" else 0.0,
        1.0 if exp.weakest_component == "EA" else 0.0,
        1.0 if exp.weakest_component == "Indirect" else 0.0,
    ]
    clf = model._clf
    probabilities = clf.predict_proba([features])[0]
    best = max(range(len(probabilities)), key=lambda i: probabilities[i])
    return str(clf.classes_[best]), round(float(probabilities[best]), 3)
