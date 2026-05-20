from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from .io_utils import normalize_keys

ID_COLUMNS = {"course_id", "semester_id", "id"}
CREDITS_COLUMNS = {"credits", "credit"}


@dataclass(frozen=True)
class CreditRow:
    id: str
    credits: float
    po_values: dict[str, float] = field(default_factory=dict)


def aggregate_by_credits(rows: list[CreditRow]) -> dict[str, float]:
    po_ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for po_id in row.po_values:
            if po_id not in seen:
                seen.add(po_id)
                po_ids.append(po_id)

    out: dict[str, float] = {}
    for po_id in po_ids:
        numerator = 0.0
        denominator = 0.0
        for row in rows:
            if po_id not in row.po_values:
                continue
            numerator += row.po_values[po_id] * row.credits
            denominator += row.credits
        out[po_id] = (numerator / denominator) if denominator else 0.0
    return out


def compute_semester_po(courses: list[CreditRow]) -> dict[str, float]:
    return aggregate_by_credits(courses)


def compute_program_po(semesters: list[CreditRow]) -> dict[str, float]:
    return aggregate_by_credits(semesters)


def _detect_columns(headers: list[str]) -> tuple[str, str, list[str]]:
    id_col: str | None = None
    credit_col: str | None = None
    po_cols: list[str] = []
    for header in headers:
        if header is None:
            continue
        key = header.strip().lower()
        if id_col is None and key in ID_COLUMNS:
            id_col = header
        elif credit_col is None and key in CREDITS_COLUMNS:
            credit_col = header
        else:
            po_cols.append(header)
    if id_col is None or credit_col is None:
        raise ValueError(
            "Input must include an id column (course_id/semester_id/id) "
            "and a credits column."
        )
    return id_col, credit_col, po_cols


def _read_tabular(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            headers = list(reader.fieldnames or [])
            rows = list(reader)
        return headers, rows
    if suffix == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, list) or not data:
            raise ValueError(f"{path.name}: JSON input must be a non-empty list of objects.")
        headers = list(data[0].keys())
        rows = [dict(item) for item in data]
        return headers, rows
    raise ValueError(f"{path.name}: unsupported extension '{suffix}'. Use .json or .csv.")


def load_credit_rows(path: str) -> list[CreditRow]:
    headers, raw_rows = _read_tabular(Path(path))
    id_col, credit_col, po_cols = _detect_columns(headers)
    id_key = id_col.strip().lower()
    credit_key = credit_col.strip().lower()
    po_keys = {col: col.strip() for col in po_cols}

    rows: list[CreditRow] = []
    for raw in raw_rows:
        normalized = normalize_keys(raw)
        po_values: dict[str, float] = {}
        for col, display in po_keys.items():
            value = normalized.get(col.strip().lower())
            if value in (None, ""):
                continue
            po_values[display] = float(value)
        rows.append(
            CreditRow(
                id=str(normalized[id_key]).strip(),
                credits=float(normalized[credit_key]),
                po_values=po_values,
            )
        )
    return rows


def write_aggregate_summary(po_values: dict[str, float], out_path: Path, level: str) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["level", "po_id", "value", "percentage", "scaled"])
        for po_id, value in po_values.items():
            writer.writerow(
                [
                    level,
                    po_id,
                    round(value, 4),
                    round(value * 100 / 3, 2),
                    round(value, 2),
                ]
            )
    return out_path


def run_semester_aggregation(courses_file: str, out_dir: str) -> Path:
    rows = load_credit_rows(courses_file)
    po_values = compute_semester_po(rows)
    out_path = Path(out_dir) / "semester_po_attainment.csv"
    return write_aggregate_summary(po_values, out_path, level="semester")


def run_program_aggregation(semesters_file: str, out_dir: str) -> Path:
    rows = load_credit_rows(semesters_file)
    po_values = compute_program_po(rows)
    out_path = Path(out_dir) / "program_po_attainment.csv"
    return write_aggregate_summary(po_values, out_path, level="program")
