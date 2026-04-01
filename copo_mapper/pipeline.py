from __future__ import annotations

import csv
import json
from pathlib import Path

from .preprocess import normalize_text
from .scoring import score_pair
from .semantic import tfidf_pair_similarity
from .types import Outcome


def _pick_value(row: dict[str, str], candidates: list[str]) -> str | None:
    lowered = {key.lower(): value for key, value in row.items()}
    for candidate in candidates:
        value = lowered.get(candidate.lower())
        if value is not None:
            return value
    return None


def _load_outcomes(path: Path) -> list[Outcome]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(path.read_text())
        return [Outcome(id=item["id"], text=item["text"]) for item in data]
    if suffix == ".csv":
        with path.open() as f:
            rows = list(csv.DictReader(f))
        outcomes: list[Outcome] = []
        for row in rows:
            outcome_id = _pick_value(row, ["id", "co", "po"])
            outcome_text = _pick_value(row, ["text", "description"])
            if outcome_id is None or outcome_text is None:
                raise ValueError(
                    "CSV must include an ID column (id/CO/PO) and a text column (text/Description)."
                )
            outcomes.append(Outcome(id=outcome_id.strip(), text=outcome_text.strip()))
        return outcomes
    raise ValueError(f"Unsupported file format for {path}. Use .json or .csv.")


def run_pairwise_mapping(co_file: str, po_file: str, out_dir: str) -> tuple[Path, Path]:
    co_items = _load_outcomes(Path(co_file))
    po_items = _load_outcomes(Path(po_file))

    rows: list[dict[str, str | int | float]] = []
    co_norms: list[str] = []
    po_norms: list[str] = []

    for co in co_items:
        for po in po_items:
            co_norm = normalize_text(co.text)
            po_norm = normalize_text(po.text)
            co_norms.append(co_norm)
            po_norms.append(po_norm)
            rows.append(
                {
                    "co_id": co.id,
                    "co_text": co.text,
                    "po_id": po.id,
                    "po_text": po.text,
                    "co_norm": co_norm,
                    "po_norm": po_norm,
                }
            )

    similarities = tfidf_pair_similarity(co_norms, po_norms)

    for i, row in enumerate(rows):
        result = score_pair(str(row["co_norm"]), str(row["po_norm"]), similarities[i])
        row["predicted_strength"] = result.score
        row["confidence"] = result.confidence
        row["explanation"] = result.explanation

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = output_dir / "pair_predictions.csv"
    matrix_path = output_dir / "matrix.csv"

    with pair_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "co_id",
                "co_text",
                "po_id",
                "po_text",
                "predicted_strength",
                "confidence",
                "explanation",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})

    po_ids = [po.id for po in po_items]
    matrix: dict[str, dict[str, int]] = {co.id: {po_id: 0 for po_id in po_ids} for co in co_items}
    for row in rows:
        matrix[str(row["co_id"])][str(row["po_id"])] = int(row["predicted_strength"])

    with matrix_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["co_id", *po_ids])
        for co in co_items:
            writer.writerow([co.id, *[matrix[co.id][po_id] for po_id in po_ids]])

    return pair_path, matrix_path
