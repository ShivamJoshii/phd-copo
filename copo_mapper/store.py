"""SQLite store for attainment runs, reasons and actions (the CQI history).

One row per outcome per offering, keyed by
``academic_year + semester + course_id + level + outcome_id``. *Every* CO/PO
of a saved run is stored (achieved ones too), because next semester's row for
the same CO is how the feedback loop checks whether an action worked.

The database is a single file (``COPO_DB_PATH`` env var, default
``copo_cqi.sqlite`` in the working directory). It survives app restarts on a
normal computer or server. Hosts with throw-away disks (e.g. Streamlit
Community Cloud) lose it on redeploy, so ``export_csv`` / ``import_csv`` give
a manual backup that carries everything.

stdlib only (sqlite3, csv).
"""

from __future__ import annotations

import csv
import os
import sqlite3
from datetime import datetime, timezone
from io import StringIO
from typing import Iterable, Sequence

DEFAULT_DB_PATH = "copo_cqi.sqlite"

KEY_COLUMNS = ["academic_year", "semester", "course_id", "level", "outcome_id"]

# name -> SQL type. Order is the CSV export order.
COLUMNS: dict[str, str] = {
    "academic_year": "TEXT NOT NULL",
    "semester": "TEXT NOT NULL",
    "course_id": "TEXT NOT NULL",
    "level": "TEXT NOT NULL",          # CO | PO
    "outcome_id": "TEXT NOT NULL",
    "course_type": "TEXT",
    "class_size": "INTEGER",
    "ma": "REAL",
    "ea": "REAL",
    "indirect": "REAL",
    "direct": "REAL",
    "final": "REAL",
    "scaled": "REAL",
    "target": "REAL",
    "gap": "REAL",
    "achieved": "INTEGER",
    "slab": "INTEGER",
    "bloom_level": "TEXT",
    "map_strength": "REAL",
    "weakest": "TEXT",
    "rule_reason": "TEXT",             # expert-tree assumption
    "ml_reason": "TEXT",               # learned-tree prediction at save time
    "faculty_reason": "TEXT",          # confirmed / overridden by faculty
    "reasoning_text": "TEXT",
    "action_code": "TEXT",             # catalogue codes, ", " joined
    "action_plan": "TEXT",
    "responsible": "TEXT",
    "review_date": "TEXT",
    "is_demo": "INTEGER",
    "saved_at": "TEXT",
}

# Faculty-entered columns: a re-save with a blank value must not erase them.
STICKY_COLUMNS = [
    "faculty_reason", "reasoning_text", "action_code", "action_plan", "responsible", "review_date",
]

_NUMERIC = {name for name, sql in COLUMNS.items() if sql.startswith(("REAL", "INTEGER"))}


def db_path() -> str:
    return os.environ.get("COPO_DB_PATH", DEFAULT_DB_PATH)


def connect(path: str | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or db_path())
    conn.row_factory = sqlite3.Row
    cols = ", ".join(f"{name} {sql}" for name, sql in COLUMNS.items())
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS cqi_records ({cols}, PRIMARY KEY ({', '.join(KEY_COLUMNS)}))"
    )
    conn.commit()
    return conn


def _clean(row: dict) -> dict:
    out: dict = {}
    for name in COLUMNS:
        value = row.get(name)
        if isinstance(value, str):
            value = value.strip()
        if value == "":
            value = None if name in _NUMERIC else ""
        if name in _NUMERIC and value is not None:
            value = float(value)
            if COLUMNS[name].startswith("INTEGER"):
                value = int(value)
        out[name] = value
    for key in KEY_COLUMNS:
        if not out[key]:
            raise ValueError(f"'{key}' is required to save a record.")
    out["saved_at"] = out["saved_at"] or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    out["is_demo"] = int(out["is_demo"] or 0)
    return out


def upsert_rows(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    """Insert or update rows by key. Blank faculty fields never erase saved ones."""
    names = list(COLUMNS)
    updates = []
    for name in names:
        if name in KEY_COLUMNS:
            continue
        if name in STICKY_COLUMNS:
            updates.append(f"{name} = COALESCE(NULLIF(excluded.{name}, ''), cqi_records.{name})")
        else:
            updates.append(f"{name} = excluded.{name}")
    sql = (
        f"INSERT INTO cqi_records ({', '.join(names)}) VALUES ({', '.join('?' for _ in names)}) "
        f"ON CONFLICT ({', '.join(KEY_COLUMNS)}) DO UPDATE SET {', '.join(updates)}"
    )
    count = 0
    for row in rows:
        clean = _clean(row)
        conn.execute(sql, [clean[name] for name in names])
        count += 1
    conn.commit()
    return count


def fetch_rows(conn: sqlite3.Connection, include_demo: bool = True) -> list[dict]:
    where = "" if include_demo else "WHERE is_demo = 0"
    cur = conn.execute(
        f"SELECT * FROM cqi_records {where} ORDER BY academic_year, semester, course_id, level, outcome_id"
    )
    return [dict(row) for row in cur.fetchall()]


def delete_demo(conn: sqlite3.Connection) -> int:
    cur = conn.execute("DELETE FROM cqi_records WHERE is_demo = 1")
    conn.commit()
    return cur.rowcount


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    cur = conn.execute(
        "SELECT COUNT(*) AS n, COALESCE(SUM(is_demo), 0) AS demo, "
        "COUNT(DISTINCT academic_year || '|' || semester || '|' || course_id) AS runs FROM cqi_records"
    )
    row = cur.fetchone()
    return {"records": row["n"], "demo": row["demo"], "real": row["n"] - row["demo"], "runs": row["runs"]}


def export_csv(rows: Sequence[dict]) -> str:
    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(COLUMNS))
    writer.writeheader()
    for row in rows:
        writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in COLUMNS})
    return buf.getvalue()


def import_csv(conn: sqlite3.Connection, text: str) -> int:
    """Load a backup made by ``export_csv`` (rows missing a key are skipped)."""
    rows = []
    for row in csv.DictReader(StringIO(text.lstrip("﻿"))):
        normalized = {(k or "").strip().lower(): v for k, v in row.items()}
        if all(str(normalized.get(k) or "").strip() for k in KEY_COLUMNS):
            rows.append(normalized)
    return upsert_rows(conn, rows)


# ---------------------------------------------------------------------------
# Building rows from a Stage 2 run
# ---------------------------------------------------------------------------


def rows_from_run(
    *,
    academic_year: str,
    semester: str,
    course_id: str,
    course_type: str,
    class_size: int | None,
    co_results,
    po_results,
    diagnosis,
    config,
    suggestions: dict,
    bloom_by_co: dict[str, str] | None = None,
    map_strength_by_co: dict[str, float] | None = None,
    ml_reasons: dict[str, str] | None = None,
    action_records: Sequence = (),
) -> list[dict]:
    """Rows for every CO and PO of one run, with any saved faculty decisions."""
    from .cqi_rules import slab  # noqa: PLC0415

    bloom_by_co = bloom_by_co or {}
    map_strength_by_co = map_strength_by_co or {}
    ml_reasons = ml_reasons or {}
    faculty = {
        (r.level, r.outcome_id): r
        for r in action_records
        if r.course_id == course_id and r.academic_year == academic_year and r.semester == semester
    }
    co_exp = {e.co_id: e for e in diagnosis.co_explanations}
    po_exp = {e.po_id: e for e in diagnosis.po_explanations}
    base = {
        "academic_year": academic_year,
        "semester": semester,
        "course_id": course_id,
        "course_type": course_type,
        "class_size": class_size,
        "is_demo": 0,
    }

    def decided(level: str, outcome_id: str) -> dict:
        rec = faculty.get((level, outcome_id))
        suggestion = suggestions.get((level, outcome_id))
        return {
            "rule_reason": suggestion.reason_id if suggestion else "",
            "faculty_reason": rec.reason if rec else "",
            "reasoning_text": rec.reasoning_text if rec else "",
            "action_code": rec.action_code if rec else "",
            "action_plan": rec.action_plan if rec else "",
            "responsible": rec.responsible if rec else "",
            "review_date": rec.review_date if rec else "",
        }

    rows: list[dict] = []
    for co in co_results:
        exp = co_exp[co.co_id]
        rows.append(
            {
                **base,
                "level": "CO",
                "outcome_id": co.co_id,
                "ma": co.ma_attainment,
                "ea": co.ea_attainment,
                "indirect": co.indirect_attainment,
                "direct": co.direct_attainment,
                "final": co.final_attainment,
                "scaled": co.scaled_attainment,
                "target": config.co_target_level,
                "gap": max(0.0, exp.gap),
                "achieved": 1 if exp.achieved else 0,
                "slab": slab(co.scaled_attainment),
                "bloom_level": bloom_by_co.get(co.co_id, ""),
                "map_strength": map_strength_by_co.get(co.co_id),
                "weakest": exp.weakest_component,
                "ml_reason": ml_reasons.get(co.co_id, ""),
                **decided("CO", co.co_id),
            }
        )
    for po in po_results:
        exp = po_exp[po.po_id]
        rows.append(
            {
                **base,
                "level": "PO",
                "outcome_id": po.po_id,
                "final": po.weighted_attainment,
                "scaled": po.scaled_attainment,
                "target": config.po_target_level,
                "gap": max(0.0, exp.gap),
                "achieved": 1 if exp.achieved else 0,
                "slab": slab(po.scaled_attainment),
                "weakest": exp.contributions[0].co_id if exp.contributions else "",
                "ml_reason": "",
                **decided("PO", po.po_id),
            }
        )
    return rows
