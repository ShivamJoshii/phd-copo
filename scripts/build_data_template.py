"""Build docs/data_collection_template.xlsx: the three datasets faculty fill in.

    python scripts/build_data_template.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copo_mapper.action_plan import REASON_TAXONOMY  # noqa: E402
from copo_mapper.cqi_rules import ACTION_CATALOG, COURSE_TYPES, REASON_CODES  # noqa: E402
from copo_mapper.features import BLOOM_ORDER  # noqa: E402

FONT = Font(name="Arial", size=10)
BOLD = Font(name="Arial", size=10, bold=True)
HEAD = Font(name="Arial", size=10, bold=True, color="FFFFFF")
HEAD_FILL = PatternFill("solid", fgColor="1F4E78")
INPUT_FILL = PatternFill("solid", fgColor="FFFF00")
EXAMPLE_FONT = Font(name="Arial", size=10, italic=True, color="7F7F7F")
ROWS = 300

KEY = [("academic_year", "e.g. 2025-26"), ("semester", "e.g. 5"), ("course_id", "Same spelling everywhere: CS301")]

SHEETS = {
    "PO_Master": (
        [("program", "e.g. B.Tech CSE"), ("po_id", "PO1..PO12, PSO1.."), ("po_text", "Full NBA statement, not just 'PO1'")],
        ["B.Tech CSE", "PO1", "Apply knowledge of mathematics, science and engineering fundamentals to solve complex engineering problems."],
    ),
    "CO_Master": (
        [("program", ""), ("regulation", "Syllabus version"), ("course_id", ""), ("course_name", ""),
         ("course_type", "Theory / Practical"), ("semester", ""), ("co_id", "CO1.."), ("co_text", "No 'CO1:' prefix"),
         ("bloom_level", "Optional")],
        ["B.Tech CSE", "R2022", "CS301", "Database Management Systems", "Theory", "5", "CO1",
         "Design normalized relational schemas for real-world applications.", "create"],
    ),
    "Mapping_Labels": (
        [("course_id", ""), ("co_id", ""), ("po_id", ""), ("level", "0/1/2/3; blank in the old matrix = 0"),
         ("approved_by", "Faculty who approved the matrix")],
        ["CS301", "CO1", "PO3", 3, "Dr. A. Sharma"],
    ),
    "Attainment_Inputs": (
        [*KEY, ("course_type", "Theory / Practical"), ("co_id", ""),
         ("internal_avg", "0-1 (e.g. 0.72 = 72%)"), ("external_avg", "0-1, end-semester"),
         ("indirect_avg", "0-1 (survey rating / 5)"), ("students", "Class size"),
         ("survey_responses", ""), ("absent_policy", "excluded / counted as zero")],
        ["2025-26", "5", "CS301", "Theory", "CO1", 0.72, 0.58, 0.80, 60, 54, "excluded"],
    ),
    "Action_Taken": (
        [*KEY, ("co_id", ""), ("attained", "Scaled 0-3"), ("target", "Scaled 0-3"),
         ("reason_code", "From Codes sheet"), ("faculty_verified", "Yes / corrected code"),
         ("action_code", "From Codes sheet"), ("action_text", "What was actually done"),
         ("responsible", "Faculty name"), ("review_date", "YYYY-MM-DD"),
         ("next_attained", "Same CO, next offering"), ("improved", "Y / N")],
        ["2025-26", "5", "CS301", "CO1", 1.94, 2.1, "R3", "Yes", "A05",
         "Internal papers moderated against end-semester standard by a second examiner.",
         "Dr. A. Sharma", "2026-12-15", 2.18, "Y"],
    ),
}

DROPDOWNS = {
    "course_type": "=Codes!$A$2:$A$3",
    "bloom_level": f"=Codes!$B$2:$B${1 + len(BLOOM_ORDER)}",
    "level": '"0,1,2,3"',
    "reason_code": f"=Codes!$D$2:$D${1 + len(REASON_CODES)}",
    "action_code": f"=Codes!$G$2:$G${1 + len(ACTION_CATALOG)}",
    "improved": '"Y,N"',
    "absent_policy": '"excluded,counted as zero"',
}


def style_header(ws, n_cols: int) -> None:
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=1, column=c)
        cell.font, cell.fill = HEAD, HEAD_FILL
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"


def main() -> None:
    wb = Workbook()
    readme = wb.active
    readme.title = "Read_Me"
    lines = [
        ("CO-PO data collection template", BOLD),
        ("Fill the yellow cells. Row 2 on every sheet is an example in grey italics: overwrite or delete it.", FONT),
        ("Use the same course code spelling on every sheet (CS301, not 'CS 301'). Key = academic_year + semester + course_id + co_id.", FONT),
        ("", FONT),
        ("PO_Master / CO_Master / Mapping_Labels: faculty-approved matrices, one row per CO-PO pair. Used to train and test the mapper (scripts/train_mapper.py).", FONT),
        ("Attainment_Inputs: one row per CO per course per semester. Averages on the 0-1 scale. Used to compute attainment.", FONT),
        ("Action_Taken: one row per CO that missed its target. Pick reason and action codes from the lists (see Codes). Fill next_attained and improved a year later.", FONT),
        ("Codes: the allowed values behind the dropdowns. Data_Dictionary: what every column means.", FONT),
        ("", FONT),
        ("Before sharing: replace student roll numbers with anonymous ids. This template holds course-level averages only.", FONT),
    ]
    for i, (text, font) in enumerate(lines, 1):
        cell = readme.cell(row=i, column=1, value=text)
        cell.font = font
        cell.alignment = Alignment(wrap_text=True, vertical="top")
    readme.column_dimensions["A"].width = 120

    dictionary = [("sheet", "column", "meaning / allowed values")]
    for name, (columns, example) in SHEETS.items():
        ws = wb.create_sheet(name)
        for c, (col, note) in enumerate(columns, 1):
            ws.cell(row=1, column=c, value=col)
            ws.column_dimensions[get_column_letter(c)].width = max(14, min(48, len(str(example[c - 1])) + 4))
            ex = ws.cell(row=2, column=c, value=example[c - 1])
            ex.font, ex.fill = EXAMPLE_FONT, INPUT_FILL
            for r in range(3, ROWS + 1):
                cell = ws.cell(row=r, column=c)
                cell.fill, cell.font = INPUT_FILL, FONT
            dictionary.append((name, col, note))
            if col in DROPDOWNS:
                dv = DataValidation(type="list", formula1=DROPDOWNS[col], allow_blank=True)
                dv.error, dv.errorTitle = "Pick a value from the list (see the Codes sheet).", "Not in list"
                ws.add_data_validation(dv)
                letter = get_column_letter(c)
                dv.add(f"{letter}2:{letter}{ROWS}")
        style_header(ws, len(columns))

    codes = wb.create_sheet("Codes")
    heads = ["course_type", "bloom_level", None, "reason_code", "reason", None, "action_code", "action", "theory wording", "practical wording"]
    for c, h in enumerate(heads, 1):
        if h:
            cell = codes.cell(row=1, column=c, value=h)
            cell.font, cell.fill = HEAD, HEAD_FILL
    for i, v in enumerate(COURSE_TYPES, 2):
        codes.cell(row=i, column=1, value=v).font = FONT
    for i, v in enumerate(BLOOM_ORDER, 2):
        codes.cell(row=i, column=2, value=v).font = FONT
    for i, (rid, code) in enumerate(REASON_CODES.items(), 2):
        codes.cell(row=i, column=4, value=code).font = FONT
        codes.cell(row=i, column=5, value=REASON_TAXONOMY[rid].label).font = FONT
    for i, a in enumerate(ACTION_CATALOG.values(), 2):
        for c, v in zip((7, 8, 9, 10), (a.code, a.label, a.theory, a.practical)):
            codes.cell(row=i, column=c, value=v).font = FONT
    for letter, width in zip("ABDEGHIJ", (14, 14, 12, 42, 12, 34, 80, 80)):
        codes.column_dimensions[letter].width = width
    codes.freeze_panes = "A2"

    dd = wb.create_sheet("Data_Dictionary")
    for r, line in enumerate(dictionary, 1):
        for c, v in enumerate(line, 1):
            cell = dd.cell(row=r, column=c, value=v)
            cell.font = FONT
    style_header(dd, 3)
    for letter, width in zip("ABC", (20, 20, 70)):
        dd.column_dimensions[letter].width = width

    out = Path(__file__).resolve().parents[1] / "docs" / "data_collection_template.xlsx"
    wb.save(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
