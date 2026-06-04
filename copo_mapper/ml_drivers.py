"""Systemic driver analysis: *across many courses*, what predicts a miss?

This is the "ML later" layer. The per-course root cause lives in
``copo_mapper.diagnostics`` (deterministic, exact). Here we look across a whole
*dataset* of past outcomes to surface patterns no single course reveals — e.g.
"misses are driven far more by external (EA) than internal (MA) assessment", or
"indirect attainment barely moves the outcome".

Design choices for small datasets (the realistic case early on):

* The default analysis is **dependency-free and interpretable** — a
  point-biserial correlation (Pearson r between each feature and the binary
  missed/met label) plus the mean-value gap between missed and met groups.
  With only a little data this is more honest than a deep model that would
  overfit, and every number is auditable.
* An optional ``fit_logistic`` upgrade path uses scikit-learn when available
  (guarded import, same pattern as the SBERT/BERT backends). It returns
  ``None`` if sklearn is not installed, so callers degrade gracefully to the
  correlation ranking.

The feature rows are produced from the same objects the diagnostics use
(``observation_from_co``), so the deterministic and ML layers share one schema.
"""

from __future__ import annotations

import importlib.util
import math
from dataclasses import dataclass

from .attainment import COAttainmentResult, WeightConfig

# Features available per CO observation. Kept explicit so the schema is stable
# across the dependency-free ranking and any future trained model.
CO_FEATURE_KEYS = [
    "ma_attainment",
    "ea_attainment",
    "indirect_attainment",
    "direct_attainment",
    "final_attainment",
]


def observation_from_co(
    co: COAttainmentResult, config: WeightConfig, *, course_id: str = ""
) -> dict[str, float | int | str]:
    """Turn one CO result into a labelled feature row for driver analysis."""
    return {
        "course_id": course_id,
        "co_id": co.co_id,
        "ma_attainment": co.ma_attainment,
        "ea_attainment": co.ea_attainment,
        "indirect_attainment": co.indirect_attainment,
        "direct_attainment": co.direct_attainment,
        "final_attainment": co.final_attainment,
        # Label: 1 == missed the target, 0 == met it.
        "missed": int(co.scaled_attainment < config.co_target_level),
    }


@dataclass(frozen=True)
class DriverScore:
    feature: str
    correlation: float        # point-biserial r with `missed` (+ => higher feature, more misses)
    mean_when_missed: float
    mean_when_met: float
    gap: float                # mean_when_met - mean_when_missed
    direction: str            # "protective" (higher => fewer misses) / "risk" / "flat"
    note: str


def _pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var_x = sum((x - mx) ** 2 for x in xs)
    var_y = sum((y - my) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return 0.0
    return cov / math.sqrt(var_x * var_y)


def rank_drivers(
    observations: list[dict],
    *,
    feature_keys: list[str] | None = None,
    label_key: str = "missed",
) -> list[DriverScore]:
    """Rank features by how strongly they associate with a missed target.

    Returns drivers sorted by absolute correlation (strongest first). A negative
    correlation means a higher feature value goes with *fewer* misses
    (protective); positive means it goes with *more* misses (risk).
    """
    keys = feature_keys or CO_FEATURE_KEYS
    labels = [float(o[label_key]) for o in observations]
    missed_idx = [i for i, v in enumerate(labels) if v == 1.0]
    met_idx = [i for i, v in enumerate(labels) if v == 0.0]

    scores: list[DriverScore] = []
    for key in keys:
        values = [float(o[key]) for o in observations]
        r = _pearson(values, labels)
        mean_missed = (sum(values[i] for i in missed_idx) / len(missed_idx)) if missed_idx else float("nan")
        mean_met = (sum(values[i] for i in met_idx) / len(met_idx)) if met_idx else float("nan")
        gap = (mean_met - mean_missed) if (missed_idx and met_idx) else float("nan")

        if r <= -0.1:
            direction = "protective"
        elif r >= 0.1:
            direction = "risk"
        else:
            direction = "flat"

        scores.append(
            DriverScore(
                feature=key,
                correlation=round(r, 4),
                mean_when_missed=round(mean_missed, 4) if not math.isnan(mean_missed) else mean_missed,
                mean_when_met=round(mean_met, 4) if not math.isnan(mean_met) else mean_met,
                gap=round(gap, 4) if not math.isnan(gap) else gap,
                direction=direction,
                note="",
            )
        )

    scores.sort(key=lambda s: abs(s.correlation), reverse=True)
    return scores


def summarize_drivers(scores: list[DriverScore], n_observations: int) -> list[str]:
    """Human-readable headline findings, with an honest small-sample caveat."""
    lines: list[str] = []
    if n_observations < 10:
        lines.append(
            f"⚠ Only {n_observations} observations — treat these as directional hints, "
            "not statistically robust drivers. Collect more before acting."
        )
    top = [s for s in scores if s.direction != "flat"]
    if not top:
        lines.append("No feature shows a clear association with misses in this dataset.")
        return lines
    for s in top[:3]:
        verb = "lower" if s.direction == "protective" else "higher"
        lines.append(
            f"{s.feature}: {verb} values track with misses (r={s.correlation:+.2f}; "
            f"mean met {s.mean_when_met:.2f} vs missed {s.mean_when_missed:.2f})."
        )
    return lines


def fit_logistic(
    observations: list[dict],
    *,
    feature_keys: list[str] | None = None,
    label_key: str = "missed",
):
    """Optional scikit-learn logistic model. Returns None if sklearn is absent.

    Provided as the upgrade path once enough labelled data exists. The returned
    object exposes ``coefficients`` (per-feature, standardised) and ``predict``.
    Callers should fall back to :func:`rank_drivers` when this returns None.
    """
    if importlib.util.find_spec("sklearn") is None:
        return None

    keys = feature_keys or CO_FEATURE_KEYS
    labels = [int(o[label_key]) for o in observations]
    if len(set(labels)) < 2:
        # Degenerate: every row is the same class, nothing to learn.
        return None

    import numpy as np  # noqa: WPS433 (local import keeps base install dependency-free)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    x = np.array([[float(o[k]) for k in keys] for o in observations])
    y = np.array(labels)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_scaled, y)

    @dataclass(frozen=True)
    class _Fitted:
        feature_keys: list[str]
        coefficients: dict[str, float]

        def predict(self, rows: list[dict]) -> list[float]:
            xs = scaler.transform([[float(r[k]) for k in keys] for r in rows])
            return [float(p) for p in model.predict_proba(xs)[:, 1]]

    return _Fitted(
        feature_keys=keys,
        coefficients={k: round(float(c), 4) for k, c in zip(keys, model.coef_[0])},
    )
