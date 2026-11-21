"""Reason for non-attainment: one plain report row per missed CO / PO.

When the final output says a CO or PO was *not* achieved, this module says
why, from the inputs that produced it:

1. **Where the marks were lost** (exact arithmetic). The required final level
   is target / 3. Each input's shortfall below that level is weighted by how
   much it counts in the final formula:

       final  = direct_w * Direct + indirect_w * Indirect
       Direct = (ma_w * Internal + ea_w * External) / (ma_w + ea_w)

       loss(Indirect) = indirect_w * max(0, required - Indirect)
       loss(Internal) = direct_w * ma_w/(ma_w+ea_w) * max(0, required - Internal)
       loss(External) = direct_w * ea_w/(ma_w+ea_w) * max(0, required - External)
       loss(Direct)   = loss(Internal) + loss(External)

   The largest loss is the main cause. Shares are reported as % of total loss.

2. **Assumed reason** — the expert decision tree in ``action_plan`` (the
   supervised decision-tree model is trained on these labels later). Every
   reason is stated with the assumption behind it and the exact branch taken,
   so reviewers can disagree with it.

3. **What would fix it** — the cheapest single-input change that reaches the
   target (from ``diagnostics``) and the suggested CQI actions.

POs are explained through the missed COs that pull them down most.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from io import StringIO
from pathlib import Path
from typing import Sequence

from .action_plan import REASON_TAXONOMY, suggest_for_course
from .attainment import WeightConfig
from .diagnostics import COExplanation, CourseDiagnosis, POExplanation


@dataclass(frozen=True)
class NonAttainmentRow:
    level: str                # "CO" or "PO"
    outcome_id: str
    attained: float           # scaled 0..3
    target: float
    gap: float
    internal: str             # component values (CO) or "" (PO)
    external: str
    direct: str
    indirect: str
    main_cause: str           # e.g. "External (end-semester) — 61% of the loss"
    loss_breakdown: str       # "Internal 0.03 (22%) · External 0.08 (61%) · Indirect 0.02 (17%)"
    assumed_reason: str       # reason label
    assumption: str           # why we assume it (taxonomy description)
    decision_path: str        # branches taken, " → " joined
    cheapest_fix: str
    suggested_actions: str    # " | " joined


ROW_FIELDS = [f.name for f in fields(NonAttainmentRow)]

COMPONENT_NAMES = {
    "MA": "Internal (MA)",
    "EA": "External (EA, end-semester)",
    "Indirect": "Indirect (course-exit survey)",
}


def co_losses(exp: COExplanation, config: WeightConfig) -> dict[str, float]:
    """Weighted shortfall of each input below the required final level."""
    required = exp.target / 3.0
    de_w = config.ma_weight + config.ea_weight
    ma_share = config.ma_weight / de_w if de_w else 0.0
    ea_share = config.ea_weight / de_w if de_w else 0.0
    c = exp.components
    return {
        "MA": config.direct_weight * ma_share * max(0.0, required - c["MA"]),
        "EA": config.direct_weight * ea_share * max(0.0, required - c["EA"]),
        "Indirect": config.indirect_weight * max(0.0, required - c["Indirect"]),
    }


def _fmt_losses(losses: dict[str, float]) -> tuple[str, str]:
    total = sum(losses.values())
    if total <= 0:
        return (
            "No single input is below the required level; the weighted mix still misses",
            "none below required level",
        )
    pct = {k: v / total * 100 for k, v in losses.items()}
    main = max(losses, key=losses.get)
    direct_pct = pct["MA"] + pct["EA"]
    side = "direct" if direct_pct >= pct["Indirect"] else "indirect"
    main_cause = (
        f"{COMPONENT_NAMES[main]} — {pct[main]:.0f}% of the loss "
        f"(direct {direct_pct:.0f}% / indirect {pct['Indirect']:.0f}%, mostly {side})"
    )
    breakdown = " · ".join(
        f"{k} {losses[k]:.3f} ({pct[k]:.0f}%)" for k in ("MA", "EA", "Indirect")
    )
    return main_cause, breakdown


def _co_row(exp: COExplanation, config: WeightConfig, suggestion, direct: float) -> NonAttainmentRow:
    main_cause, breakdown = _fmt_losses(co_losses(exp, config))
    reason = REASON_TAXONOMY[suggestion.reason_id] if suggestion else None
    fix = (
        f"Raise {COMPONENT_NAMES[exp.levers[0].name]} {exp.levers[0].current:.2f} → "
        f"{exp.levers[0].needed:.2f}"
        if exp.levers
        else "No single input can reach target alone; improve several together"
    )
    c = exp.components
    return NonAttainmentRow(
        level="CO",
        outcome_id=exp.co_id,
        attained=exp.scaled,
        target=exp.target,
        gap=exp.gap,
        internal=f"{c['MA']:.2f}",
        external=f"{c['EA']:.2f}",
        direct=f"{direct:.2f}",
        indirect=f"{c['Indirect']:.2f}",
        main_cause=main_cause,
        loss_breakdown=breakdown,
        assumed_reason=reason.label if reason else "",
        assumption=reason.description if reason else "",
        decision_path=" → ".join(suggestion.decision_path) if suggestion else "",
        cheapest_fix=fix,
        suggested_actions=" | ".join(suggestion.actions) if suggestion else "",
    )


def _po_row(exp: POExplanation, suggestion, missed_co_ids: set[str]) -> NonAttainmentRow:
    draggers = [c for c in exp.contributions if c.drag_scaled > 0]
    total_drag = sum(c.drag_scaled for c in draggers)
    if draggers and total_drag > 0:
        top = draggers[0]
        missed = "missed" if top.co_id in missed_co_ids else "met its own target"
        main_cause = (
            f"{top.co_id} ({missed}) — {top.drag_scaled / total_drag * 100:.0f}% of the drag "
            f"(final {top.co_final:.2f}, mapping weight {top.weight_share * 100:.0f}%)"
        )
        breakdown = " · ".join(
            f"{c.co_id} {c.drag_scaled:.3f} ({c.drag_scaled / total_drag * 100:.0f}%)" for c in draggers
        )
    elif not exp.contributions:
        main_cause, breakdown = "No CO maps to this PO", ""
    else:
        main_cause, breakdown = "Mapped COs are at target; the mapping weights sit on weaker COs", ""
    reason = REASON_TAXONOMY[suggestion.reason_id] if suggestion else None
    fix = (
        f"Raise {exp.levers[0].name} final {exp.levers[0].current:.2f} → {exp.levers[0].needed:.2f}"
        if exp.levers
        else "No single CO can reach target alone; improve several COs"
    )
    return NonAttainmentRow(
        level="PO",
        outcome_id=exp.po_id,
        attained=exp.scaled,
        target=exp.target,
        gap=exp.gap,
        internal="",
        external="",
        direct="",
        indirect="",
        main_cause=main_cause,
        loss_breakdown=breakdown,
        assumed_reason=reason.label if reason else "",
        assumption=reason.description if reason else "",
        decision_path=" → ".join(suggestion.decision_path) if suggestion else "",
        cheapest_fix=fix,
        suggested_actions=" | ".join(suggestion.actions) if suggestion else "",
    )


def build_non_attainment_report(
    diagnosis: CourseDiagnosis,
    config: WeightConfig,
    direct_by_co: dict[str, float],
) -> list[NonAttainmentRow]:
    """One row per missed CO, then per missed PO. Empty when everything is met."""
    suggestions = suggest_for_course(diagnosis)
    rows = [
        _co_row(exp, config, suggestions.get(("CO", exp.co_id)), direct_by_co.get(exp.co_id, 0.0))
        for exp in diagnosis.missed_cos
    ]
    missed_ids = {e.co_id for e in diagnosis.missed_cos}
    rows += [
        _po_row(exp, suggestions.get(("PO", exp.po_id)), missed_ids) for exp in diagnosis.missed_pos
    ]
    return rows


def report_to_csv(rows: Sequence[NonAttainmentRow]) -> str:
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=ROW_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow(asdict(row))
    return buf.getvalue()


def write_report(rows: Sequence[NonAttainmentRow], path: Path) -> Path:
    path.write_text(report_to_csv(rows), encoding="utf-8")
    return path
