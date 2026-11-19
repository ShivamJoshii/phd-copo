"""Faculty review of the predicted CO-PO matrix.

The scorer proposes a strength for every CO-PO cell; faculty / reviewers may
change any cell and must say why. This module keeps that review separate from
the prediction so both stay auditable:

* the **predicted** matrix is never modified,
* each change is a ``MappingOverride`` (predicted -> final, category, reason),
* the **reviewed** matrix = predicted + overrides, and is what Stage 2 uses.

Overrides round-trip through CSV so a review can be saved and re-applied to a
later run, and they double as labelled data for a future trained mapper.

Pure Python, no Streamlit import, so it is unit-testable.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from io import StringIO
from typing import Iterable, Sequence

OVERRIDE_CATEGORIES: tuple[str, ...] = (
    "Stronger link than predicted",
    "Weaker link than predicted",
    "Not addressed in course delivery",
    "Directly assessed in course",
    "Programme committee decision",
    "Other",
)

VALID_STRENGTHS = (0, 1, 2, 3)

Matrix = dict[str, dict[str, int]]  # co_id -> po_id -> strength


@dataclass(frozen=True)
class MappingOverride:
    co_id: str
    po_id: str
    predicted_strength: int
    final_strength: int
    category: str
    reason: str
    reviewer: str = ""
    timestamp: str = ""


OVERRIDE_FIELDS = [f.name for f in fields(MappingOverride)]


# ---------------------------------------------------------------------------
# Matrix helpers (header/rows form used by the UI <-> dict form)
# ---------------------------------------------------------------------------


def matrix_from_rows(header: Sequence[str], rows: Iterable[Sequence[object]]) -> Matrix:
    po_ids = list(header[1:])
    return {str(row[0]): {po: int(row[i + 1]) for i, po in enumerate(po_ids)} for row in rows}


def matrix_to_rows(matrix: Matrix, po_ids: Sequence[str]) -> list[list[str]]:
    return [[co_id, *[str(cells[po]) for po in po_ids]] for co_id, cells in matrix.items()]


def diff_matrices(predicted: Matrix, edited: Matrix) -> list[tuple[str, str, int, int]]:
    """Cells where ``edited`` differs from ``predicted``: (co, po, old, new)."""
    out: list[tuple[str, str, int, int]] = []
    for co_id, cells in predicted.items():
        for po_id, old in cells.items():
            new = edited.get(co_id, {}).get(po_id, old)
            if int(new) != int(old):
                out.append((co_id, po_id, int(old), int(new)))
    return out


def apply_overrides(predicted: Matrix, overrides: Iterable[MappingOverride]) -> Matrix:
    """Reviewed matrix. Overrides for unknown cells are ignored."""
    reviewed = {co: dict(cells) for co, cells in predicted.items()}
    for ov in overrides:
        if ov.co_id in reviewed and ov.po_id in reviewed[ov.co_id]:
            reviewed[ov.co_id][ov.po_id] = int(ov.final_strength)
    return reviewed


def build_overrides(
    predicted: Matrix,
    edited: Matrix,
    reasons: dict[tuple[str, str], tuple[str, str]],
    reviewer: str = "",
    now: str | None = None,
    previous: Iterable[MappingOverride] = (),
) -> list[MappingOverride]:
    """Turn a matrix edit into override records.

    ``reasons`` maps (co_id, po_id) -> (category, free-text reason).
    Raises ValueError listing every changed cell that has no written reason,
    so an unexplained change can never reach Stage 2. Records in ``previous``
    whose strength, category and reason are unchanged are kept as-is, so
    re-saving never rewrites who made an earlier decision or when.
    """
    prior = {(o.co_id, o.po_id): o for o in previous}
    stamp = now or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    changes = diff_matrices(predicted, edited)
    missing = [f"{co}→{po}" for co, po, _, _ in changes if not reasons.get((co, po), ("", ""))[1].strip()]
    if missing:
        raise ValueError("A reason is required for every changed cell: " + ", ".join(missing))
    out: list[MappingOverride] = []
    for co, po, old, new in changes:
        if new not in VALID_STRENGTHS:
            raise ValueError(f"{co}→{po}: strength must be 0, 1, 2 or 3 (got {new}).")
        category, reason = reasons[(co, po)]
        kept = prior.get((co, po))
        if (
            kept is not None
            and kept.final_strength == new
            and kept.category == (category or "Other")
            and kept.reason == reason.strip()
        ):
            out.append(kept)
            continue
        out.append(
            MappingOverride(
                co_id=co,
                po_id=po,
                predicted_strength=old,
                final_strength=new,
                category=category or "Other",
                reason=reason.strip(),
                reviewer=reviewer.strip(),
                timestamp=stamp,
            )
        )
    return out


# ---------------------------------------------------------------------------
# CSV round-trip
# ---------------------------------------------------------------------------


def overrides_to_csv(overrides: Sequence[MappingOverride]) -> str:
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=OVERRIDE_FIELDS)
    writer.writeheader()
    for ov in overrides:
        writer.writerow(asdict(ov))
    return buf.getvalue()


def overrides_from_csv(text: str) -> list[MappingOverride]:
    out: list[MappingOverride] = []
    for row in csv.DictReader(StringIO(text.lstrip("﻿"))):
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        if not r.get("co_id") or not r.get("po_id"):
            continue
        final = int(r["final_strength"])
        if final not in VALID_STRENGTHS:
            raise ValueError(f"{r['co_id']}→{r['po_id']}: final_strength must be 0-3.")
        out.append(
            MappingOverride(
                co_id=r["co_id"],
                po_id=r["po_id"],
                predicted_strength=int(r.get("predicted_strength") or 0),
                final_strength=final,
                category=r.get("category") or "Other",
                reason=r.get("reason", ""),
                reviewer=r.get("reviewer", ""),
                timestamp=r.get("timestamp", ""),
            )
        )
    return out


def annotate_pair_rows(
    pair_rows: Sequence[dict[str, str]], overrides: Sequence[MappingOverride]
) -> list[dict[str, str]]:
    """Pair rows with final_strength + faculty override columns added."""
    by_cell = {(o.co_id, o.po_id): o for o in overrides}
    out: list[dict[str, str]] = []
    for row in pair_rows:
        ov = by_cell.get((row["co_id"], row["po_id"]))
        new = dict(row)
        new["final_strength"] = str(ov.final_strength) if ov else str(row["predicted_strength"])
        new["faculty_override"] = "Y" if ov else "N"
        new["override_category"] = ov.category if ov else ""
        new["override_reason"] = ov.reason if ov else ""
        new["reviewer"] = ov.reviewer if ov else ""
        out.append(new)
    return out
