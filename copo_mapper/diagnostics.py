"""Root-cause diagnostics for missed CO / PO attainment targets.

The attainment pipeline is fully deterministic (weighted averages), so the
reason a target was missed is *exactly computable* rather than something to be
guessed by a model. This module decomposes a miss into:

* which input component (MA / EA / Indirect) or which CO is the weak link, and
* the precise value a single lever would need to reach in order to cross target
  (the cheapest realistic fix).

These structured explanations feed the Streamlit "Why did targets miss?" panel
and can later become labelled features for the systemic ML driver model
(see ``copo_mapper.ml_drivers``).
"""

from __future__ import annotations

from dataclasses import dataclass

from .attainment import (
    COAttainmentResult,
    POAttainmentResult,
    WeightConfig,
    compute_direct_attainment,
    compute_final_attainment,
)


@dataclass(frozen=True)
class Lever:
    """A single actionable change that would move an outcome to its target."""

    name: str          # e.g. "EA" for a CO, or a CO id for a PO
    current: float     # current value of the lever (0..1 final/component scale)
    needed: float      # value required to hit target, holding others fixed
    delta: float       # needed - current
    feasible: bool     # True if `needed` stays within [0, 1]
    note: str


@dataclass(frozen=True)
class COExplanation:
    co_id: str
    scaled: float
    target: float
    gap: float                       # target - scaled (positive => missed)
    achieved: bool
    components: dict[str, float]     # {"MA":.., "EA":.., "Indirect":..}
    weakest_component: str
    levers: list[Lever]              # feasible single-component fixes, cheapest first
    reason: str


@dataclass(frozen=True)
class POContribution:
    co_id: str
    map_strength: int
    weight_share: float              # map_strength / sum(map_strength)
    co_final: float                  # CO final attainment (0..1)
    contribution_scaled: float       # co_final * weight_share * 3
    drag_scaled: float               # how far below "at-target" this CO pulls PO


@dataclass(frozen=True)
class POExplanation:
    po_id: str
    scaled: float
    target: float
    gap: float
    achieved: bool
    contributions: list[POContribution]   # sorted by drag, worst first
    levers: list[Lever]                    # per-CO fixes, cheapest first
    reason: str


@dataclass(frozen=True)
class CourseDiagnosis:
    co_explanations: list[COExplanation]
    po_explanations: list[POExplanation]

    @property
    def missed_cos(self) -> list[COExplanation]:
        return [e for e in self.co_explanations if not e.achieved]

    @property
    def missed_pos(self) -> list[POExplanation]:
        return [e for e in self.po_explanations if not e.achieved]


def _clamp01(value: float) -> bool:
    return 0.0 <= value <= 1.0


def diagnose_co(co: COAttainmentResult, config: WeightConfig) -> COExplanation:
    """Explain a single CO's attainment against the CO target level."""
    target = config.co_target_level
    scaled = co.scaled_attainment
    gap = round(target - scaled, 4)
    achieved = scaled >= target

    components = {
        "MA": co.ma_attainment,
        "EA": co.ea_attainment,
        "Indirect": co.indirect_attainment,
    }
    weakest = min(components, key=components.get)

    # final attainment required to hit the (scaled) target.
    needed_final = target / 3.0
    ma_w, ea_w = config.ma_weight, config.ea_weight
    de_w = ma_w + ea_w
    dir_w, ind_w = config.direct_weight, config.indirect_weight

    levers: list[Lever] = []
    if not achieved:
        # Each lever solves final(component=x)=needed_final, holding others fixed.
        # Indirect: final = direct*dir_w + x*ind_w
        if ind_w > 0:
            x = (needed_final - co.direct_attainment * dir_w) / ind_w
            levers.append(_make_lever("Indirect", co.indirect_attainment, x))
        # MA / EA both feed `direct`; back out the required direct first.
        if dir_w > 0 and de_w > 0:
            needed_direct = (needed_final - co.indirect_attainment * ind_w) / dir_w
            if ma_w > 0:
                x = (needed_direct * de_w - co.ea_attainment * ea_w) / ma_w
                levers.append(_make_lever("MA", co.ma_attainment, x))
            if ea_w > 0:
                x = (needed_direct * de_w - co.ma_attainment * ma_w) / ea_w
                levers.append(_make_lever("EA", co.ea_attainment, x))

    feasible = sorted((lv for lv in levers if lv.feasible), key=lambda lv: lv.delta)

    if achieved:
        reason = f"{co.co_id} met target ({scaled:.2f} ≥ {target:.2f})."
    else:
        bits = [
            f"{co.co_id} scaled {scaled:.2f} < target {target:.2f} (gap {gap:.2f}).",
            f"Weakest input: {weakest} ({components[weakest]:.2f}).",
        ]
        if feasible:
            lv = feasible[0]
            bits.append(
                f"Cheapest fix: raise {lv.name} {lv.current:.2f} → {lv.needed:.2f}."
            )
        else:
            bits.append("No single input change can reach target; improve several together.")
        reason = " ".join(bits)

    return COExplanation(
        co_id=co.co_id,
        scaled=scaled,
        target=target,
        gap=gap,
        achieved=achieved,
        components=components,
        weakest_component=weakest,
        levers=feasible,
        reason=reason,
    )


def _make_lever(name: str, current: float, needed: float) -> Lever:
    needed_r = round(needed, 4)
    return Lever(
        name=name,
        current=round(current, 4),
        needed=needed_r,
        delta=round(needed - current, 4),
        feasible=_clamp01(needed_r),
        note="" if _clamp01(needed_r) else "out of [0,1] range",
    )


def diagnose_po(
    po: POAttainmentResult,
    co_results: list[COAttainmentResult],
    mapping: dict[str, dict[str, int]],
    config: WeightConfig,
) -> POExplanation:
    """Explain a single PO's attainment, attributing it across contributing COs."""
    target = config.po_target_level
    scaled = po.scaled_attainment
    gap = round(target - scaled, 4)
    achieved = scaled >= target

    co_final = {row.co_id: row.final_attainment for row in co_results}
    strengths: dict[str, int] = {}
    for co_id, row in mapping.items():
        s = int(row.get(po.po_id, 0) or 0)
        if s > 0:
            strengths[co_id] = s
    total = sum(strengths.values())
    weighted = po.weighted_attainment           # = scaled / 3
    needed_weighted = target / 3.0

    contributions: list[POContribution] = []
    levers: list[Lever] = []
    for co_id, strength in strengths.items():
        share = strength / total if total else 0.0
        final = co_final.get(co_id, 0.0)
        contributions.append(
            POContribution(
                co_id=co_id,
                map_strength=strength,
                weight_share=round(share, 4),
                co_final=round(final, 4),
                contribution_scaled=round(final * share * 3, 4),
                drag_scaled=round(max(0.0, needed_weighted - final) * share * 3, 4),
            )
        )
        if not achieved and share > 0:
            # Raise this CO's final so the PO weighted average hits target.
            new_final = final + (needed_weighted - weighted) / share
            levers.append(_make_lever(co_id, final, new_final))

    contributions.sort(key=lambda c: c.drag_scaled, reverse=True)
    feasible = sorted((lv for lv in levers if lv.feasible), key=lambda lv: lv.delta)

    if achieved:
        reason = f"{po.po_id} met target ({scaled:.2f} ≥ {target:.2f})."
    elif not contributions:
        reason = f"{po.po_id} scaled {scaled:.2f} < target {target:.2f}; no CO maps to it."
    else:
        worst = contributions[0]
        bits = [
            f"{po.po_id} scaled {scaled:.2f} < target {target:.2f} (gap {gap:.2f}).",
            f"Largest drag: {worst.co_id} "
            f"(final {worst.co_final:.2f}, weight {worst.weight_share*100:.0f}%).",
        ]
        if feasible:
            lv = feasible[0]
            bits.append(
                f"Cheapest fix: lift {lv.name} final {lv.current:.2f} → {lv.needed:.2f}."
            )
        else:
            bits.append("No single CO can reach target alone; improve several COs.")
        reason = " ".join(bits)

    return POExplanation(
        po_id=po.po_id,
        scaled=scaled,
        target=target,
        gap=gap,
        achieved=achieved,
        contributions=contributions,
        levers=feasible,
        reason=reason,
    )


def diagnose_course(
    co_results: list[COAttainmentResult],
    po_results: list[POAttainmentResult],
    mapping: dict[str, dict[str, int]],
    config: WeightConfig,
) -> CourseDiagnosis:
    """Full course diagnosis: an explanation for every CO and PO."""
    return CourseDiagnosis(
        co_explanations=[diagnose_co(co, config) for co in co_results],
        po_explanations=[diagnose_po(po, co_results, mapping, config) for po in po_results],
    )
