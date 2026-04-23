from __future__ import annotations

import csv
import json
from pathlib import Path

from .preprocess import normalize_text
from .scoring import score_pair
from .semantic import tfidf_pair_similarity
from .types import Outcome

CO_ID_KEY = "CO"
CO_TEXT_KEY = "description"
PO_ID_KEY = "PO"
PO_TEXT_KEY = "description"


def _load_outcomes(path: Path, id_key: str, text_key: str) -> list[Outcome]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="") as f:
            rows: list[dict[str, str]] = list(csv.DictReader(f))
    elif suffix == ".json":
        rows = json.loads(path.read_text())
        if not isinstance(rows, list):
            raise ValueError(f"{path.name}: JSON input must be a list of objects.")
    else:
        raise ValueError(f"{path.name}: unsupported extension '{suffix}'. Use .json or .csv.")

    outcomes: list[Outcome] = []
    for item in rows:
        if id_key not in item or text_key not in item:
            raise ValueError(
                f"{path.name}: each row must include columns '{id_key}' and '{text_key}'."
            )
        outcomes.append(Outcome(id=str(item[id_key]).strip(), text=str(item[text_key]).strip()))
    return outcomes


def run_pairwise_mapping(co_file: str, po_file: str, out_dir: str) -> tuple[Path, Path]:
    co_items = _load_outcomes(Path(co_file), id_key=CO_ID_KEY, text_key=CO_TEXT_KEY)
    po_items = _load_outcomes(Path(po_file), id_key=PO_ID_KEY, text_key=PO_TEXT_KEY)

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
