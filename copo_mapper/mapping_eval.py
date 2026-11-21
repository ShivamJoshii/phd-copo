"""Evaluate a CO-PO mapper against faculty-approved mappings.

Labels are the correlation levels 0-3. Three things are reported, to stay
comparable with the published work:

* **accuracy** and **macro-F1** (macro, because level 0 dominates);
* **quadratic weighted Cohen's kappa**: predicting 2 where faculty gave 3 is a
  smaller error than predicting 0, and kappa also discounts chance agreement;
* a **confusion matrix**.

``group_split`` keeps every pair of a course on one side of the split. A
random pair-level split puts the same CO text in both train and test and
inflates accuracy.

Pure Python, so it runs anywhere (the app, tests, a Colab notebook).
"""

from __future__ import annotations

import csv
import random
from io import StringIO
from typing import Sequence

LEVELS = (0, 1, 2, 3)
LABEL_FIELDS = ["course_id", "co_id", "co_text", "po_id", "po_text", "level"]


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int]) -> list[list[int]]:
    """matrix[true][pred] over levels 0-3."""
    matrix = [[0] * len(LEVELS) for _ in LEVELS]
    for t, p in zip(y_true, y_pred):
        matrix[int(t)][int(p)] += 1
    return matrix


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return sum(int(t) == int(p) for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Mean F1 over the levels that occur in truth or prediction."""
    m = confusion_matrix(y_true, y_pred)
    scores = []
    for k in range(len(LEVELS)):
        tp = m[k][k]
        fp = sum(m[i][k] for i in range(len(LEVELS))) - tp
        fn = sum(m[k]) - tp
        if tp + fp + fn == 0:
            continue
        scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores) if scores else 0.0


def weighted_kappa(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Quadratic weighted Cohen's kappa (1 = perfect, 0 = chance)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    k = len(LEVELS)
    m = confusion_matrix(y_true, y_pred)
    row = [sum(m[i]) for i in range(k)]
    col = [sum(m[i][j] for i in range(k)) for j in range(k)]
    observed = expected = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            observed += w * m[i][j]
            expected += w * row[i] * col[j] / n
    return 1.0 if expected == 0 else 1 - observed / expected


def evaluate(y_true: Sequence[int], y_pred: Sequence[int]) -> dict:
    return {
        "n": len(y_true),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        "weighted_kappa": round(weighted_kappa(y_true, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }


def group_split(
    rows: Sequence[dict], test_fraction: float = 0.2, seed: int = 0, group_key: str = "course_id"
) -> tuple[list[dict], list[dict]]:
    """Split so that every row of a course lands on the same side."""
    groups = sorted({str(r[group_key]) for r in rows})
    random.Random(seed).shuffle(groups)
    n_test = max(1, round(len(groups) * test_fraction)) if len(groups) > 1 else 0
    test_groups = set(groups[:n_test])
    train = [r for r in rows if str(r[group_key]) not in test_groups]
    test = [r for r in rows if str(r[group_key]) in test_groups]
    return train, test


def load_labelled_pairs(text: str) -> list[dict]:
    """Long-format labels: course_id, co_id, co_text, po_id, po_text, level (0-3)."""
    out = []
    for row in csv.DictReader(StringIO(text.lstrip("﻿"))):
        r = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        if not r.get("co_text") or not r.get("po_text"):
            continue
        level = int(float(r.get("level") or 0))
        if level not in LEVELS:
            raise ValueError(f"{r.get('course_id')} {r.get('co_id')}-{r.get('po_id')}: level must be 0-3.")
        out.append({**{k: r.get(k, "") for k in LABEL_FIELDS}, "level": level})
    return out


def labelled_pairs_from_review(pair_rows: Sequence[dict], course_id: str) -> list[dict]:
    """Training labels from an app review: annotated pair rows -> long format.

    ``pair_rows`` are the app's exported pair predictions (they carry
    ``final_strength`` after faculty review of the matrix).
    """
    return [
        {
            "course_id": course_id,
            "co_id": r["co_id"],
            "co_text": r["co_text"],
            "po_id": r["po_id"],
            "po_text": r["po_text"],
            "level": int(r.get("final_strength") or r["predicted_strength"]),
        }
        for r in pair_rows
    ]
