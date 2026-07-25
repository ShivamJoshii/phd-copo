"""Build the visual user guide PDF for the CO-PO Attainment Workbench.

Run from repo root:
    python docs/build_user_guide.py
Produces docs/user_guide.pdf.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUT = Path(__file__).parent / "user_guide.pdf"
PAGE_W, PAGE_H = LETTER
MARGIN = 0.6 * inch

PRIMARY = colors.HexColor("#1f4e79")
ACCENT = colors.HexColor("#2e75b6")
LIGHT = colors.HexColor("#deebf7")
GREY = colors.HexColor("#595959")
SOFT = colors.HexColor("#f2f2f2")

STRENGTH_COLORS = {
    0: colors.HexColor("#f8d7da"),
    1: colors.HexColor("#fff3cd"),
    2: colors.HexColor("#d1ecf1"),
    3: colors.HexColor("#d4edda"),
}


styles = getSampleStyleSheet()
H1 = ParagraphStyle(
    "H1", parent=styles["Heading1"], fontSize=22, textColor=PRIMARY, spaceAfter=10
)
H2 = ParagraphStyle(
    "H2", parent=styles["Heading2"], fontSize=16, textColor=PRIMARY, spaceAfter=6, spaceBefore=14
)
H3 = ParagraphStyle(
    "H3", parent=styles["Heading3"], fontSize=12.5, textColor=ACCENT, spaceAfter=4, spaceBefore=10
)
BODY = ParagraphStyle(
    "Body", parent=styles["BodyText"], fontSize=10.5, leading=15, spaceAfter=6
)
CAPTION = ParagraphStyle(
    "Caption", parent=BODY, fontSize=9, textColor=GREY, alignment=TA_CENTER, spaceAfter=10
)
CELL = ParagraphStyle(
    "Cell", parent=BODY, fontSize=9.5, leading=12, spaceAfter=0
)
CELL_HEAD = ParagraphStyle(
    "CellHead",
    parent=BODY,
    fontSize=9.5,
    leading=12,
    spaceAfter=0,
    textColor=colors.white,
    fontName="Helvetica-Bold",
)
MONO = ParagraphStyle(
    "Mono",
    parent=BODY,
    fontName="Courier",
    fontSize=9,
    leading=12,
    leftIndent=10,
    backColor=SOFT,
    borderColor=colors.HexColor("#d0d0d0"),
    borderWidth=0.5,
    borderPadding=6,
    spaceAfter=8,
)
COVER_TITLE = ParagraphStyle(
    "CoverTitle",
    parent=styles["Title"],
    fontSize=30,
    leading=36,
    textColor=PRIMARY,
    alignment=TA_CENTER,
    spaceAfter=14,
)
COVER_SUB = ParagraphStyle(
    "CoverSub",
    parent=BODY,
    fontSize=14,
    alignment=TA_CENTER,
    textColor=GREY,
    spaceAfter=20,
)


# --- Drawing flowables ---

class FlowDiagram(Flowable):
    """Vertical funnel diagram for the OBE pipeline."""

    def __init__(self, width: float, height: float):
        super().__init__()
        self.width = width
        self.height = height

    def wrap(self, *_):
        return self.width, self.height

    def draw(self):
        c = self.canv
        steps = [
            ("Internal + External", "raw assessment scores", "#fce4d6"),
            ("Direct Attainment", "(Int·w + Ext·w) / (w+w)", "#fff2cc"),
            ("Final CO", "Direct·w + Indirect·w", "#e2efda"),
            ("Course PO", "Σ(FinalCO · Map) / Σ Map", "#deebf7"),
            ("Semester PO", "Σ(CoursePO · credits) / Σ credits", "#d9d2e9"),
            ("Program PO", "Σ(SemPO · credits) / Σ credits", "#ead1dc"),
        ]
        n = len(steps)
        gap = 6
        box_h = (self.height - gap * (n - 1)) / n
        box_w = self.width - 40
        x = 20
        for i, (title, sub, fill) in enumerate(steps):
            y = self.height - (i + 1) * box_h - i * gap
            c.setFillColor(colors.HexColor(fill))
            c.setStrokeColor(colors.HexColor("#888"))
            c.setLineWidth(0.7)
            c.roundRect(x, y, box_w, box_h, 6, fill=1, stroke=1)
            c.setFillColor(colors.HexColor("#222"))
            c.setFont("Helvetica-Bold", 11)
            c.drawString(x + 12, y + box_h - 16, title)
            c.setFont("Helvetica", 9)
            c.setFillColor(colors.HexColor("#444"))
            c.drawString(x + 12, y + 6, sub)
            # arrow to next
            if i < n - 1:
                arr_x = x + box_w / 2
                arr_top = y
                arr_bot = y - gap
                c.setStrokeColor(colors.HexColor("#666"))
                c.setLineWidth(1.2)
                c.line(arr_x, arr_top, arr_x, arr_bot + 2)
                # arrowhead
                c.setFillColor(colors.HexColor("#666"))
                p = c.beginPath()
                p.moveTo(arr_x - 4, arr_bot + 3)
                p.lineTo(arr_x + 4, arr_bot + 3)
                p.lineTo(arr_x, arr_bot - 2)
                p.close()
                c.drawPath(p, fill=1, stroke=0)


class UIMockup(Flowable):
    """Mockup of the Streamlit page: title bar, tab strip, sidebar, main content."""

    def __init__(self, width: float, height: float, active_step: int, content_lines: list[str]):
        super().__init__()
        self.width = width
        self.height = height
        self.active_step = active_step
        self.content_lines = content_lines

    def wrap(self, *_):
        return self.width, self.height

    def draw(self):
        c = self.canv
        # Outer browser frame
        c.setStrokeColor(colors.HexColor("#bbb"))
        c.setLineWidth(0.8)
        c.setFillColor(colors.white)
        c.rect(0, 0, self.width, self.height, fill=1, stroke=1)
        # Title bar
        title_h = 22
        c.setFillColor(PRIMARY)
        c.rect(0, self.height - title_h, self.width, title_h, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(10, self.height - 15, "CO-PO Mapping & Attainment Workbench")
        # Sidebar
        side_w = 110
        side_top = self.height - title_h
        side_h = self.height - title_h
        c.setFillColor(LIGHT)
        c.rect(0, 0, side_w, side_h, fill=1, stroke=0)
        c.setFont("Helvetica-Bold", 8)
        c.setFillColor(PRIMARY)
        c.drawString(8, side_top - 14, f"Step {self.active_step} Inputs")
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.HexColor("#333"))
        sidebar_items = [
            "[ Upload file ]",
            "[ Upload file ]",
            "Backend: tfidf v",
            "Model: (default)",
        ]
        for i, item in enumerate(sidebar_items):
            c.drawString(8, side_top - 30 - i * 14, item)
        # Tab strip
        tab_top = side_top - 8
        tab_h = 18
        tab_w = (self.width - side_w - 20) / 4
        tabs = ["Step 1: Mapping", "Step 2: Course", "Step 3: Semester", "Step 4: Program"]
        for i, label in enumerate(tabs):
            x = side_w + 10 + i * tab_w
            active = (i + 1) == self.active_step
            c.setFillColor(PRIMARY if active else SOFT)
            c.rect(x, tab_top - tab_h, tab_w - 2, tab_h, fill=1, stroke=0)
            c.setFillColor(colors.white if active else colors.HexColor("#333"))
            c.setFont("Helvetica-Bold" if active else "Helvetica", 7.5)
            c.drawString(x + 6, tab_top - tab_h + 6, label)
        # Main content
        cx = side_w + 10
        cy = tab_top - tab_h - 14
        c.setFillColor(colors.HexColor("#222"))
        c.setFont("Helvetica", 9)
        for line in self.content_lines:
            if line.startswith("#"):
                c.setFont("Helvetica-Bold", 9.5)
                c.setFillColor(PRIMARY)
                c.drawString(cx, cy, line[1:].strip())
                cy -= 14
                c.setFillColor(colors.HexColor("#222"))
                c.setFont("Helvetica", 9)
            elif line.startswith("BTN:"):
                txt = line[4:].strip()
                w = c.stringWidth(txt, "Helvetica-Bold", 8.5) + 16
                c.setFillColor(ACCENT)
                c.roundRect(cx, cy - 4, w, 18, 3, fill=1, stroke=0)
                c.setFillColor(colors.white)
                c.setFont("Helvetica-Bold", 8.5)
                c.drawString(cx + 8, cy + 2, txt)
                c.setFillColor(colors.HexColor("#222"))
                c.setFont("Helvetica", 9)
                cy -= 24
            elif line.startswith("BOX:"):
                txt = line[4:].strip()
                c.setStrokeColor(colors.HexColor("#aaa"))
                c.setFillColor(colors.HexColor("#fafafa"))
                c.rect(cx, cy - 4, self.width - cx - 14, 18, fill=1, stroke=1)
                c.setFillColor(colors.HexColor("#444"))
                c.drawString(cx + 6, cy + 2, txt)
                c.setFillColor(colors.HexColor("#222"))
                cy -= 24
            else:
                c.drawString(cx, cy, line)
                cy -= 13
            if cy < 10:
                break


class MatrixSample(Flowable):
    """Color-coded sample CO×PO matrix."""

    def __init__(self, width: float):
        super().__init__()
        self.width = width
        self.height = 110

    def wrap(self, *_):
        return self.width, self.height

    def draw(self):
        c = self.canv
        cols = ["CO\\PO", "PO1", "PO2", "PO3", "PO4"]
        rows = [
            ("CO1", [3, 2, 0, 1]),
            ("CO2", [2, 3, 1, 0]),
            ("CO3", [1, 2, 3, 2]),
        ]
        cell_w = self.width / len(cols)
        cell_h = 22
        # header
        c.setFillColor(PRIMARY)
        c.rect(0, self.height - cell_h, self.width, cell_h, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 9.5)
        for i, label in enumerate(cols):
            c.drawCentredString(i * cell_w + cell_w / 2, self.height - cell_h + 7, label)
        # rows
        for r, (rid, vals) in enumerate(rows):
            y = self.height - cell_h - (r + 1) * cell_h
            c.setFillColor(SOFT)
            c.rect(0, y, cell_w, cell_h, fill=1, stroke=0)
            c.setFillColor(colors.HexColor("#222"))
            c.setFont("Helvetica-Bold", 9)
            c.drawCentredString(cell_w / 2, y + 7, rid)
            for i, v in enumerate(vals):
                cx = (i + 1) * cell_w
                c.setFillColor(STRENGTH_COLORS[v])
                c.rect(cx, y, cell_w, cell_h, fill=1, stroke=0)
                c.setFillColor(colors.HexColor("#222"))
                c.setFont("Helvetica", 9.5)
                c.drawCentredString(cx + cell_w / 2, y + 7, str(v))
        c.setStrokeColor(colors.HexColor("#aaa"))
        c.setLineWidth(0.4)
        for i in range(len(cols) + 1):
            c.line(i * cell_w, 0, i * cell_w, self.height)
        for j in range(len(rows) + 2):
            c.line(0, j * cell_h, self.width, j * cell_h)


# --- Helpers ---

def code(text: str) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>").replace(" ", "&nbsp;"), MONO)


def schema_table(rows: list[tuple[str, str]], headers: tuple[str, str] = ("Field", "Meaning")) -> Table:
    data = [[Paragraph(headers[0], CELL_HEAD), Paragraph(headers[1], CELL_HEAD)]]
    for a, b in rows:
        data.append([Paragraph(a, CELL), Paragraph(b, CELL)])
    t = Table(data, colWidths=[1.9 * inch, 4.4 * inch])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SOFT]),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbb")),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#ddd")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return t


# --- Page builders ---

def cover() -> list:
    return [
        Spacer(1, 1.4 * inch),
        Paragraph("CO-PO Attainment Workbench", COVER_TITLE),
        Paragraph("Visual User Guide", COVER_SUB),
        Spacer(1, 0.3 * inch),
        FlowDiagram(width=4.5 * inch, height=4.5 * inch),
        Spacer(1, 0.4 * inch),
        Paragraph(
            "A step-by-step walkthrough of Stages 1 through 4 — from CO/PO mapping all the way "
            "to program-level outcome attainment.",
            CAPTION,
        ),
        PageBreak(),
    ]


def overview() -> list:
    return [
        Paragraph("1. What this tool does", H1),
        Paragraph(
            "The workbench takes course assessment data and turns it into the four reports an "
            "OBE-accredited program needs:",
            BODY,
        ),
        Paragraph(
            "<b>Final CO Attainment</b> · <b>Course-wise PO Attainment</b> · "
            "<b>Semester-wise PO Attainment</b> · <b>Program-level PO Attainment</b>",
            BODY,
        ),
        Paragraph(
            "Every stage is the same shape: a <b>weighted average</b>. Only the weights change "
            "as you climb the funnel.",
            BODY,
        ),
        Spacer(1, 8),
        Paragraph("The funnel at a glance", H2),
        FlowDiagram(width=5.5 * inch, height=4.6 * inch),
        Spacer(1, 4),
        Paragraph(
            "Each box's output is the next box's input. The Streamlit app mirrors this with "
            "four steps you walk left → right.",
            CAPTION,
        ),
        PageBreak(),
    ]


def getting_started() -> list:
    return [
        Paragraph("2. Getting started", H1),
        Paragraph("Option A — Streamlit Cloud", H2),
        Paragraph(
            "Point Streamlit Cloud at the repo and let it auto-install. The included "
            "<font face='Courier'>requirements.txt</font> pulls in streamlit, sentence-transformers, "
            "transformers, and torch so all backends work out of the box.",
            BODY,
        ),
        schema_table(
            [
                ("Repository", "ShivamJoshii/phd-copo"),
                ("Branch", "main (post-merge) — or the feature branch on PR #27"),
                ("Main file", "streamlit_app.py"),
                ("Python", "3.10 or 3.11"),
                ("Secrets (optional)", "HF_TOKEN — avoids HF Hub rate limits on cold start"),
            ]
        ),
        Spacer(1, 10),
        Paragraph("Option B — Local", H2),
        code(
            "python -m venv .venv\n"
            "source .venv/bin/activate\n"
            "pip install -e \".[ui,nlp]\"\n"
            "streamlit run streamlit_app.py"
        ),
        Paragraph(
            "The app opens at <font face='Courier'>http://localhost:8501</font>. You'll see four "
            "tabs across the top — that's the flow.",
            BODY,
        ),
        Spacer(1, 8),
        Paragraph("Sample data lives in <font face='Courier'>examples/</font>", H3),
        schema_table(
            [
                ("co.json / po.json", "Step 1 mapping inputs"),
                ("co_attainment_spec.json", "Step 2 CO scores (spec worked example)"),
                ("attainment_config_spec.json", "Step 2 weights (30/50, 80/20)"),
                ("mapping_matrix_spec.csv", "Step 2 CO×PO map (worked example)"),
                ("courses_semester1.csv", "Step 3 — two courses, DBMS + OS"),
                ("semesters_program.csv", "Step 4 — two semesters"),
            ]
        ),
        PageBreak(),
    ]


def step1() -> list:
    return [
        Paragraph("3. Step 1 — CO-PO Mapping", H1),
        Paragraph("What this step produces", H2),
        Paragraph(
            "A pairwise mapping matrix: for every Course Outcome × Program Outcome pair, a "
            "strength score on a 0/1/2/3 scale.",
            BODY,
        ),
        UIMockup(
            width=6.4 * inch,
            height=2.8 * inch,
            active_step=1,
            content_lines=[
                "#Step 1 — CO-PO Mapping",
                "Upload CO + PO files (JSON or CSV) in the sidebar.",
                "Pick semantic backend: tfidf / sbert / bert.",
                "BTN:Run Mapping",
                "BOX: matrix preview appears here (color-coded 0–3) ",
            ],
        ),
        Paragraph("UI mockup of Step 1", CAPTION),
        Paragraph("Inputs", H3),
        schema_table(
            [
                ("CO file", "JSON or CSV with columns: CO, description"),
                ("PO file", "JSON or CSV with columns: PO, description (PSOs allowed as PSO1, PSO2…)"),
                ("Semantic backend", "tfidf (default) · sbert · bert"),
            ]
        ),
        Spacer(1, 8),
        Paragraph("Sample output — the mapping matrix", H3),
        MatrixSample(width=4.5 * inch),
        Spacer(1, 4),
        Paragraph(
            "Color scale — <font backColor='#f8d7da'>&nbsp;0&nbsp;</font> none · "
            "<font backColor='#fff3cd'>&nbsp;1&nbsp;</font> weak · "
            "<font backColor='#d1ecf1'>&nbsp;2&nbsp;</font> medium · "
            "<font backColor='#d4edda'>&nbsp;3&nbsp;</font> strong.",
            CAPTION,
        ),
        Paragraph("Downloads", H3),
        Paragraph(
            "<b>pair_predictions.csv</b> — every CO×PO pair with strength, confidence, "
            "explanation, and which semantic method was used.<br/>"
            "<b>matrix.csv</b> — the CO×PO grid only (this is what flows to Step 2).",
            BODY,
        ),
        PageBreak(),
    ]


def step2() -> list:
    return [
        Paragraph("4. Step 2 — Course Attainment", H1),
        Paragraph(
            "Combine assessment scores with the mapping matrix to compute final CO attainment "
            "and course-wise PO attainment.",
            BODY,
        ),
        UIMockup(
            width=6.4 * inch,
            height=3.0 * inch,
            active_step=2,
            content_lines=[
                "#Step 2 — Course Attainment",
                "CO table is seeded from the Step 1 matrix.",
                "Enter MA / EA / Indirect per CO (editable cells).",
                "Set MA weight, Direct weight, Target level.",
                "BTN:Run Attainment Analysis",
                "BOX: CO summary + PO summary tables render below",
                "BTN:Add course to Step 3",
            ],
        ),
        Paragraph("UI mockup of Step 2", CAPTION),
        Paragraph("The formulas", H2),
        code(
            "Direct  = (MA · w_ma + EA · w_ea) / (w_ma + w_ea)\n"
            "FinalCO = Direct · w_direct + Indirect · w_indirect\n"
            "PO      = Σ(FinalCO_i · Map_ij) / Σ Map_ij"
        ),
        Paragraph(
            "The Direct formula is <b>normalized</b> by the sum of internal+external weights, "
            "so a 30/50 split produces results on the same scale as the raw assessments. "
            "The legacy 40/60 (sums to 1.0) case is unchanged.",
            BODY,
        ),
        Paragraph("Worked example (matches the included spec fixtures)", H3),
        schema_table(
            [
                ("Weights", "Internal=30%, External=50%, Direct=80%, Indirect=20%"),
                ("CO1 inputs", "MA=2.5, EA=2.8, Indirect=2.7"),
                ("CO1 Direct", "(2.5·0.3 + 2.8·0.5) / 0.8 = 2.6875"),
                ("CO1 Final", "2.6875·0.8 + 2.7·0.2 = 2.69"),
                ("PO1 course", "(2.69·3 + 2.46·2 + 2.81·1) / 6 = 2.63"),
            ],
            headers=("Quantity", "Calculation"),
        ),
        Paragraph("Push forward", H3),
        Paragraph(
            "After running, the <b>Add course to Step 3</b> button appears. Type the course id "
            "(e.g. <i>DBMS</i>) and credits (e.g. <i>4</i>), click it, and Step 3 picks up the "
            "course's per-PO row automatically — no re-uploading.",
            BODY,
        ),
        PageBreak(),
    ]


def step3() -> list:
    return [
        Paragraph("5. Step 3 — Semester PO Attainment", H1),
        Paragraph(
            "Roll multiple courses up to a single semester score, weighted by course credits.",
            BODY,
        ),
        UIMockup(
            width=6.4 * inch,
            height=2.8 * inch,
            active_step=3,
            content_lines=[
                "#Step 3 — Semester PO Attainment",
                "Courses pushed from Step 2 appear in the table.",
                "Edit rows or upload courses_semester1.csv.",
                "BTN:Run Semester Aggregation",
                "BOX: Semester PO table + download button",
                "BTN:Add semester to Step 4",
            ],
        ),
        Paragraph("UI mockup of Step 3", CAPTION),
        Paragraph("Formula", H2),
        code("PO_sem = Σ(PO_course · credits) / Σ credits"),
        Paragraph("Input shape", H3),
        code(
            "course_id,credits,PO1,PO2,PO3\n"
            "DBMS,4,2.63,2.63,2.72\n"
            "OS,3,2.50,2.40,2.60"
        ),
        Paragraph(
            "Empty cells are <b>skipped</b> (treated as 'this course doesn't contribute to that "
            "PO') — not as zeros that drag the average down.",
            BODY,
        ),
        Paragraph("Worked example", H3),
        schema_table(
            [
                ("PO1", "(2.63·4 + 2.50·3) / 7 = 2.57"),
                ("PO2", "(2.63·4 + 2.40·3) / 7 = 2.53"),
                ("PO3", "(2.72·4 + 2.60·3) / 7 = 2.67"),
            ],
            headers=("Quantity", "Calculation"),
        ),
        PageBreak(),
    ]


def step4() -> list:
    return [
        Paragraph("6. Step 4 — Program PO Attainment", H1),
        Paragraph(
            "Final stage. Roll every semester up to a single program-level outcome score, "
            "credit-weighted across semesters.",
            BODY,
        ),
        UIMockup(
            width=6.4 * inch,
            height=2.8 * inch,
            active_step=4,
            content_lines=[
                "#Step 4 — Program / Degree PO Attainment",
                "Semesters pushed from Step 3 appear in the table.",
                "Edit rows or upload semesters_program.csv.",
                "BTN:Run Program Aggregation",
                "BOX: Program PO table + download button",
            ],
        ),
        Paragraph("UI mockup of Step 4", CAPTION),
        Paragraph("Formula", H2),
        code("PO_program = Σ(PO_sem · credits) / Σ credits"),
        Paragraph("Input shape", H3),
        code(
            "semester_id,credits,PO1\n"
            "Sem1,20,2.50\n"
            "Sem2,22,2.58"
        ),
        Paragraph("Worked example", H3),
        schema_table(
            [
                ("PO1 program", "(2.50·20 + 2.58·22) / 42 = 2.54"),
            ],
            headers=("Quantity", "Calculation"),
        ),
        Paragraph(
            "That's it — the value you'd report for PO1 across the whole program.",
            BODY,
        ),
        PageBreak(),
    ]


def cli_reference() -> list:
    return [
        Paragraph("7. CLI reference (no UI required)", H1),
        Paragraph(
            "Every step has a command-line entry point. Outputs are CSVs / JSON written to the "
            "<font face='Courier'>--out-dir</font> you pick.",
            BODY,
        ),
        Paragraph("Step 1 — Mapping", H3),
        code(
            "copo-map \\\n"
            "  --co-file examples/co.json \\\n"
            "  --po-file examples/po.json \\\n"
            "  --out-dir outputs"
        ),
        Paragraph("Step 2 — Course attainment", H3),
        code(
            "copo-attainment \\\n"
            "  --co-attainment-file examples/co_attainment_spec.json \\\n"
            "  --mapping-matrix-file examples/mapping_matrix_spec.csv \\\n"
            "  --config-file examples/attainment_config_spec.json \\\n"
            "  --out-dir attainment_outputs"
        ),
        Paragraph("Step 3 — Semester", H3),
        code(
            "copo-semester \\\n"
            "  --courses-file examples/courses_semester1.csv \\\n"
            "  --out-dir semester_outputs"
        ),
        Paragraph("Step 4 — Program", H3),
        code(
            "copo-program \\\n"
            "  --semesters-file examples/semesters_program.csv \\\n"
            "  --out-dir program_outputs"
        ),
        Spacer(1, 8),
        Paragraph("Verify the full pipeline at once", H2),
        Paragraph(
            "Run the four commands in order; the spec example should produce these numbers:",
            BODY,
        ),
        schema_table(
            [
                ("CO1 Direct", "2.6875 (rounded 2.69)"),
                ("CO1 Final", "2.69"),
                ("Course PO1 / PO2 / PO3", "2.63 / 2.63 / 2.72"),
                ("Semester PO1 / PO2 / PO3", "2.57 / 2.53 / 2.67"),
                ("Program PO1", "2.54"),
            ],
            headers=("Quantity", "Expected value"),
        ),
        PageBreak(),
    ]


def troubleshooting() -> list:
    return [
        Paragraph("8. Troubleshooting", H1),
        schema_table(
            [
                (
                    "RuntimeError on SBERT/BERT",
                    "The model couldn't load. Install nlp deps: <font face='Courier'>pip install -e \".[nlp]\"</font>. No silent fallback by design — this is so method comparisons stay fair.",
                ),
                (
                    "HF Hub rate limits",
                    "Set the <font face='Courier'>HF_TOKEN</font> secret in Streamlit Cloud. Unauthenticated downloads still work but throttle on cold start.",
                ),
                (
                    "Stage 2 numbers look 3× too big",
                    "The CO summary's <font face='Courier'>scaled_attainment</font> multiplies by 3 assuming 0–1 inputs. If you pass 0–3 scale inputs (like the spec), read the <font face='Courier'>final_attainment</font> column instead.",
                ),
                (
                    "Step 3/4 missing a PO",
                    "Aggregation skips empty cells. If a course intentionally doesn't touch a PO, leave the cell blank — it won't drag the average down.",
                ),
                (
                    "Streamlit cold start is slow",
                    "First boot installs torch (~700 MB). Allow 3–5 minutes on first deploy; subsequent loads are fast.",
                ),
            ],
            headers=("Symptom", "What to do"),
        ),
        Spacer(1, 12),
        Paragraph("Key rule to remember", H2),
        Paragraph(
            "<b>Every step is a weighted average.</b> Only the weights change — internal/external "
            "split, then direct/indirect, then mapping strength, then credits, then credits again.",
            BODY,
        ),
    ]


def build() -> None:
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=LETTER,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title="CO-PO Attainment Workbench — Visual User Guide",
        author="phd-copo",
    )
    story: list = []
    story += cover()
    story += overview()
    story += getting_started()
    story += step1()
    story += step2()
    story += step3()
    story += step4()
    story += cli_reference()
    story += troubleshooting()
    doc.build(story, onFirstPage=_page_chrome, onLaterPages=_page_chrome)
    print(f"Wrote {OUT}")


def _page_chrome(canvas, doc):
    canvas.saveState()
    # footer
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(GREY)
    canvas.drawString(MARGIN, 0.35 * inch, "CO-PO Attainment Workbench")
    canvas.drawRightString(
        PAGE_W - MARGIN, 0.35 * inch, f"Page {doc.page}"
    )
    # accent rule
    canvas.setStrokeColor(LIGHT)
    canvas.setLineWidth(0.6)
    canvas.line(MARGIN, 0.5 * inch, PAGE_W - MARGIN, 0.5 * inch)
    canvas.restoreState()


if __name__ == "__main__":
    build()
