"""Excel-parity tests against the NBA accreditation workbook.

Reference workbook: ``Compiler_6A(NBA)_final.xlsx`` (course: Compiler Design,
4 COs), sheet ``TOTAL ATTAINMENT``. All full-precision reference numbers below
were read directly from that sheet with openpyxl (data_only=True):

    row 10  Internal Attainment (MA)       C10:F10
    row 11  End-Semester Attainment (EA)   C11:F11
    row 12  Direct Attainment (0.4MA+0.6EA) C12:F12
    row 17  Indirect Attainment            C17:F17
    row 18  Final Attainment (0.8D+0.2I)   C18:F18
    row 19  Final * 3 (scale of 3)         C19:F19
    row 20  Target level 1.4 achieved      C20:F20  (all "Y")
    rows 26-29  CO-PO mapping matrix       C26:N29
    row 39  PO weighted attainment         C39:N39

--------------------------------------------------------------------------
KNOWN EXCEL QUIRKS (documented here on purpose, deliberately NOT replicated)
--------------------------------------------------------------------------
(a) Row-39 formula bug: the workbook's PO attainment formula is, e.g. C39:

        =(($B$35*C35)+($B$36*C36)+($B$37*C37)+($B$38))/SUM(C35,C36,C37,C38)

    The CO4 term appears as ``+$B$38`` instead of ``+$B$38*C38`` -- CO4's
    mapping-strength multiplier is missing from the numerator (while C38 IS
    still included in the denominator). Consequences in the workbook:

      * PO1/PO2/PO3/PSO2 (all weights 3): Excel shows 0.5961050443361393,
        because it computes (3*f1 + 3*f2 + 3*f3 + f4)/12 instead of the
        correct (3*f1 + 3*f2 + 3*f3 + 3*f4)/12 = mean(f) = 0.7075605349...
      * PO4/PSO3 (CO4 weight is 0): Excel shows 0.7948067257815191, because
        it computes (3*f1 + 3*f2 + 3*f3 + f4)/9 -- CO4 leaks into the
        numerator even though its weight is 0. Correct value is
        (f1 + f2 + f3)/3 = 0.7205030653...

(b) Cells I39 (PO9) and J39 (PO10) are not driven by the row formula at all:
    I39 is the hardcoded-constant formula ``=(0.6745+0.7303)/2`` (rounded
    finals typed in by hand -> 0.7024) and J39 is the literal number 0.6745.
    These happen to be numerically right to 4dp only because they were typed
    from the rounded final attainments.

The library (copo_mapper.attainment.compute_po_attainment) implements the
CORRECT intended math: weighted = sum(final_CO * strength)/sum(strength) per
PO column. Therefore the PO tests below assert the correct math, and the
workbook-verification test intentionally does NOT compare row 39 for the
columns affected by bugs (a)/(b).
"""

import csv
import json
import os
from pathlib import Path

import pytest

from copo_mapper.attainment import (
    COAttainmentInput,
    WeightConfig,
    compute_co_attainment,
    compute_direct_attainment,
    compute_final_attainment,
    compute_po_attainment,
    run_attainment_analysis_from_objects,
)

# Path used when tests run inside the Cowork sandbox (bash). Overridable so
# the suite can also run on a host checkout where the workbook lives elsewhere.
WORKBOOK_PATH = os.environ.get(
    "COPO_REFERENCE_XLSX",
    "/sessions/elegant-friendly-thompson/mnt/Claude--Projects/phd-work/"
    "Compiler_6A(NBA)_final.xlsx",
)

CO_IDS = ["CO1", "CO2", "CO3", "CO4"]

# Full-precision values from TOTAL ATTAINMENT rows 10/11/17 (inputs).
MA = {
    "CO1": 0.7786283891547051,
    "CO2": 0.8402903811252268,
    "CO3": 0.8511796733212342,
    "CO4": 0.7816764132553603,
}
EA = {
    "CO1": 0.7255639097744361,
    "CO2": 0.5248538011695906,
    "CO3": 0.6425438596491229,
    "CO4": 0.5562865497076022,
}
INDIRECT = {
    "CO1": 0.7964912280701755,
    "CO2": 0.7684210526315789,
    "CO3": 0.7473684210526316,
    "CO4": 0.7578947368421052,
}

# Excel row 12: Direct = 0.4*MA + 0.6*EA (full precision cached cell values).
EXCEL_DIRECT = {
    "CO1": 0.7467897015265437,
    "CO2": 0.651028433151845,
    "CO3": 0.7259981851179674,
    "CO4": 0.6464424951267054,
}

# Excel row 18: Final = 0.8*Direct + 0.2*Indirect (full precision).
EXCEL_FINAL = {
    "CO1": 0.75673000683527,
    "CO2": 0.6745069570477917,
    "CO3": 0.7302722323049002,
    "CO4": 0.6687329434697854,
}

# Excel row 19: Final * 3.
EXCEL_SCALED = {
    "CO1": 2.27019002050581,
    "CO2": 2.023520871143375,
    "CO3": 2.190816696914701,
    "CO4": 2.0061988304093563,
}

# CO-PO mapping matrix, TOTAL ATTAINMENT rows 26-29 (identical copy in 35-38).
PO_IDS = [
    "PO1", "PO2", "PO3", "PO4", "PO7", "PO8",
    "PO9", "PO10", "PO11", "PSO2", "PSO3", "PSO4",
]
MAPPING_ROWS = {
    "CO1": [3, 3, 3, 3, 0, 0, 0, 0, 0, 3, 3, 0],
    "CO2": [3, 3, 3, 3, 0, 0, 1, 1, 0, 3, 3, 0],
    "CO3": [3, 3, 3, 3, 0, 0, 1, 0, 0, 3, 3, 0],
    "CO4": [3, 3, 3, 0, 0, 0, 0, 0, 0, 3, 0, 0],
}
MAPPING = {co: dict(zip(PO_IDS, row)) for co, row in MAPPING_ROWS.items()}

CONFIG = WeightConfig(
    ma_weight=0.4,
    ea_weight=0.6,
    direct_weight=0.8,
    indirect_weight=0.2,
    co_target_level=1.4,
    po_target_level=1.4,
)

CO_INPUTS = [
    COAttainmentInput(co, MA[co], EA[co], INDIRECT[co]) for co in CO_IDS
]


def co_results():
    return compute_co_attainment(CO_INPUTS, CONFIG)


# ---------------------------------------------------------------------------
# CO attainment: full-precision helpers vs Excel
# ---------------------------------------------------------------------------

class TestDirectAttainmentFullPrecision:
    @pytest.mark.parametrize("co", CO_IDS)
    def test_direct_matches_excel_row12(self, co):
        # Excel C12:F12 formula: =(0.4*C10 + 0.6*C11)
        direct = compute_direct_attainment(MA[co], EA[co], CONFIG)
        assert direct == pytest.approx(EXCEL_DIRECT[co], abs=1e-12)

    @pytest.mark.parametrize("co", CO_IDS)
    def test_final_matches_excel_row18(self, co):
        # Excel C18:F18 formula: =(C16*0.8 + C17*0.2)
        direct = compute_direct_attainment(MA[co], EA[co], CONFIG)
        final = compute_final_attainment(direct, INDIRECT[co], CONFIG)
        assert final == pytest.approx(EXCEL_FINAL[co], abs=1e-12)

    @pytest.mark.parametrize("co", CO_IDS)
    def test_scaled_matches_excel_row19(self, co):
        direct = compute_direct_attainment(MA[co], EA[co], CONFIG)
        final = compute_final_attainment(direct, INDIRECT[co], CONFIG)
        assert final * 3 == pytest.approx(EXCEL_SCALED[co], abs=1e-12)


# ---------------------------------------------------------------------------
# CO attainment: pipeline results (note the library's documented rounding:
# direct/final rounded to 4dp, scaled rounded to 2dp)
# ---------------------------------------------------------------------------

class TestComputeCOAttainment:
    def test_direct_rounded_to_4dp(self):
        results = {r.co_id: r for r in co_results()}
        for co in CO_IDS:
            assert results[co].direct_attainment == pytest.approx(
                round(EXCEL_DIRECT[co], 4), abs=1e-9
            )
        # Spot-check literal values: 0.7467897.. -> 0.7468, etc.
        assert results["CO1"].direct_attainment == 0.7468
        assert results["CO2"].direct_attainment == 0.651
        assert results["CO3"].direct_attainment == 0.726
        assert results["CO4"].direct_attainment == 0.6464

    def test_final_rounded_to_4dp(self):
        results = {r.co_id: r for r in co_results()}
        for co in CO_IDS:
            assert results[co].final_attainment == pytest.approx(
                round(EXCEL_FINAL[co], 4), abs=1e-9
            )
        assert results["CO1"].final_attainment == 0.7567
        assert results["CO2"].final_attainment == 0.6745
        assert results["CO3"].final_attainment == 0.7303
        assert results["CO4"].final_attainment == 0.6687

    def test_scaled_rounded_to_2dp(self):
        results = {r.co_id: r for r in co_results()}
        expected = {"CO1": 2.27, "CO2": 2.02, "CO3": 2.19, "CO4": 2.01}
        for co in CO_IDS:
            assert results[co].scaled_attainment == pytest.approx(
                expected[co], abs=1e-9
            )

    def test_all_cos_achieve_target_1_4(self):
        # Excel row 20 shows Y for all four COs at target level 1.4.
        for r in co_results():
            assert r.target_achieved == "Y"
            assert r.scaled_attainment >= CONFIG.co_target_level


# ---------------------------------------------------------------------------
# PO attainment: correct intended math (NOT the workbook's buggy row 39 --
# see module docstring, quirks (a) and (b))
# ---------------------------------------------------------------------------

class TestComputePOAttainment:
    @pytest.fixture()
    def po_map(self):
        return {r.po_id: r for r in compute_po_attainment(co_results(), MAPPING, CONFIG)}

    def test_all_po_columns_present_no_crash(self, po_map):
        assert list(po_map.keys()) == PO_IDS

    def test_zero_weight_columns_are_zero(self, po_map):
        # PO7, PO8, PO11, PSO4 have strength 0 for every CO: denominator is 0,
        # the implementation must return 0.0 without dividing by zero.
        for po in ["PO7", "PO8", "PO11", "PSO4"]:
            assert po_map[po].weighted_attainment == 0.0
            assert po_map[po].percentage == 0.0
            assert po_map[po].scaled_attainment == 0.0
            assert po_map[po].target_achieved == "N"  # 0 < 1.4

    def test_all_three_columns_equal_mean_of_finals(self, po_map):
        # PO1, PO2, PO3, PSO2 map every CO with strength 3, so
        # weighted = (3*f1 + 3*f2 + 3*f3 + 3*f4) / 12 = mean(finals).
        # The library computes POs from the 4dp-rounded finals:
        #   (0.7567 + 0.6745 + 0.7303 + 0.6687) / 4 = 2.8302 / 4 = 0.70755
        # Full-precision intended value: 0.7075605349144369.
        # NOTE: Excel C39/D39/E39/L39 show 0.5961050443361393 instead --
        # quirk (a): the formula drops CO4's *3 multiplier, computing
        # (3*f1 + 3*f2 + 3*f3 + f4)/12. We assert the correct math.
        mean_rounded_finals = (0.7567 + 0.6745 + 0.7303 + 0.6687) / 4
        assert mean_rounded_finals == pytest.approx(0.70755, abs=1e-12)
        for po in ["PO1", "PO2", "PO3", "PSO2"]:
            assert po_map[po].weighted_attainment == pytest.approx(
                mean_rounded_finals, abs=1e-4
            )
            # Also within a rounding step of the full-precision intended value.
            assert po_map[po].weighted_attainment == pytest.approx(
                0.7075605349144369, abs=1e-4
            )

    def test_po4_hand_computed(self, po_map):
        # PO4 weights: CO1=3, CO2=3, CO3=3, CO4=0.
        # weighted = (3*0.7567 + 3*0.6745 + 3*0.7303 + 0*0.6687) / (3+3+3+0)
        #          = 3*(0.7567 + 0.6745 + 0.7303) / 9
        #          = 3 * 2.1615 / 9 = 6.4845 / 9 = 0.7205
        # NOTE: Excel F39 shows 0.7948067257815191 -- quirk (a): CO4's final
        # leaks unweighted into the numerator, (3*(f1+f2+f3)+f4)/9. The
        # correct value is (f1+f2+f3)/3 = 0.7205030653959873.
        assert po_map["PO4"].weighted_attainment == pytest.approx(0.7205, abs=1e-4)
        assert po_map["PO4"].scaled_attainment == pytest.approx(2.16, abs=0.01)
        # PSO3 has identical weights (3,3,3,0) so it must match PO4.
        assert po_map["PSO3"].weighted_attainment == po_map["PO4"].weighted_attainment

    def test_po9_hand_computed(self, po_map):
        # PO9 weights: CO1=0, CO2=1, CO3=1, CO4=0.
        # weighted = (1*0.6745 + 1*0.7303) / (1+1) = 1.4048 / 2 = 0.7024
        # NOTE: Excel I39 is the hardcoded formula =(0.6745+0.7303)/2 --
        # quirk (b) -- which happens to agree because it was typed from the
        # rounded finals. Full-precision intended value: 0.702389594676346.
        assert po_map["PO9"].weighted_attainment == pytest.approx(0.7024, abs=1e-4)
        assert po_map["PO9"].scaled_attainment == pytest.approx(2.11, abs=0.01)

    def test_po10_hand_computed(self, po_map):
        # PO10 weights: CO1=0, CO2=1, CO3=0, CO4=0.
        # weighted = 1*0.6745 / 1 = 0.6745 (CO2's final alone).
        # NOTE: Excel J39 is the hardcoded literal 0.6745 -- quirk (b).
        # Full-precision intended value: 0.6745069570477917.
        assert po_map["PO10"].weighted_attainment == pytest.approx(0.6745, abs=1e-4)
        assert po_map["PO10"].scaled_attainment == pytest.approx(2.02, abs=0.01)

    def test_po_target_flags_at_1_4(self, po_map):
        expected_y = {"PO1", "PO2", "PO3", "PO4", "PO9", "PO10", "PSO2", "PSO3"}
        for po in PO_IDS:
            expected = "Y" if po in expected_y else "N"
            assert po_map[po].target_achieved == expected, po
            # Flag must be consistent with the scaled value vs the target.
            if expected == "Y":
                assert po_map[po].scaled_attainment >= CONFIG.po_target_level
            else:
                assert po_map[po].scaled_attainment < CONFIG.po_target_level


# ---------------------------------------------------------------------------
# Direct verification against the workbook file (skipped when absent)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not Path(WORKBOOK_PATH).is_file(),
    reason=f"reference workbook not found at {WORKBOOK_PATH} "
    "(set COPO_REFERENCE_XLSX to override)",
)
def test_against_workbook_cached_values():
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.load_workbook(WORKBOOK_PATH, data_only=True)
    ws = wb["TOTAL ATTAINMENT"]
    cols = {"CO1": "C", "CO2": "D", "CO3": "E", "CO4": "F"}

    # If the workbook was saved without a cached calculation chain the
    # data_only values are None -- skip gracefully in that case.
    if ws["C12"].value is None:
        pytest.skip("workbook has no cached formula values (data_only=True)")

    for co, col in cols.items():
        ma = ws[f"{col}10"].value
        ea = ws[f"{col}11"].value
        indirect = ws[f"{col}17"].value

        # Inputs in this test file must equal the workbook's inputs.
        assert ma == pytest.approx(MA[co], abs=1e-14)
        assert ea == pytest.approx(EA[co], abs=1e-14)
        assert indirect == pytest.approx(INDIRECT[co], abs=1e-14)

        # Library reproduces Excel's Direct (row 12) and Final (row 18/19).
        direct = compute_direct_attainment(ma, ea, CONFIG)
        final = compute_final_attainment(direct, indirect, CONFIG)
        assert direct == pytest.approx(ws[f"{col}12"].value, abs=1e-12)
        assert final == pytest.approx(ws[f"{col}18"].value, abs=1e-12)
        assert final * 3 == pytest.approx(ws[f"{col}19"].value, abs=1e-12)
        assert ws[f"{col}20"].value == "Y"

    # Mapping matrix in the workbook (rows 26-29) matches MAPPING.
    matrix_cols = "CDEFGHIJKLMN"
    for row_idx, co in zip((26, 27, 28, 29), CO_IDS):
        for col_letter, po in zip(matrix_cols, PO_IDS):
            assert ws[f"{col_letter}{row_idx}"].value == MAPPING[co][po], (co, po)

    # Row 39 is intentionally NOT asserted against the library for
    # PO1/PO2/PO3/PSO2 (C/D/E/L39) and PO4/PSO3 (F/M39): Excel's formula bug
    # (quirk (a)) makes those cells wrong. Instead, verify the workbook cells
    # equal the value its *buggy* formula produces, proving the bug analysis:
    #   buggy = (3*f1 + 3*f2 + 3*f3 + f4) / sum(weights)
    f = [ws[f"{c}18"].value for c in "CDEF"]
    buggy_num = 3 * f[0] + 3 * f[1] + 3 * f[2] + f[3]  # CO4 weight missing
    assert ws["C39"].value == pytest.approx(buggy_num / 12, abs=1e-12)  # PO1
    assert ws["F39"].value == pytest.approx(buggy_num / 9, abs=1e-12)   # PO4
    # ...and that they differ from the correct math by more than rounding.
    correct_po1 = sum(f) / 4
    correct_po4 = (f[0] + f[1] + f[2]) / 3
    assert abs(ws["C39"].value - correct_po1) > 0.05
    assert abs(ws["F39"].value - correct_po4) > 0.05
    # Quirk (b): I39/J39 are hardcoded from 4dp-rounded finals; they agree
    # with the correct math only to ~4dp.
    assert ws["I39"].value == pytest.approx((0.6745 + 0.7303) / 2, abs=1e-12)
    assert ws["J39"].value == pytest.approx(0.6745, abs=1e-12)


# ---------------------------------------------------------------------------
# Golden end-to-end run through run_attainment_analysis_from_objects
# ---------------------------------------------------------------------------

def test_golden_end_to_end(tmp_path):
    paths = run_attainment_analysis_from_objects(
        CO_INPUTS, MAPPING, CONFIG, str(tmp_path / "out")
    )

    with paths["co_summary"].open() as fh:
        co_rows = {row["co_id"]: row for row in csv.DictReader(fh)}
    assert set(co_rows) == set(CO_IDS)
    expected_co = {
        # co_id: (direct 4dp, final 4dp, scaled 2dp, flag)
        "CO1": ("0.7468", "0.7567", "2.27", "Y"),
        "CO2": ("0.651", "0.6745", "2.02", "Y"),
        "CO3": ("0.726", "0.7303", "2.19", "Y"),
        "CO4": ("0.6464", "0.6687", "2.01", "Y"),
    }
    for co, (direct, final, scaled, flag) in expected_co.items():
        assert co_rows[co]["direct_attainment"] == direct
        assert co_rows[co]["final_attainment"] == final
        assert co_rows[co]["scaled_attainment"] == scaled
        assert co_rows[co]["target_achieved"] == flag

    with paths["po_summary"].open() as fh:
        po_rows = {row["po_id"]: row for row in csv.DictReader(fh)}
    assert list(po_rows) == PO_IDS
    expected_po_weighted = {
        "PO1": 0.7076, "PO2": 0.7076, "PO3": 0.7076, "PO4": 0.7205,
        "PO7": 0.0, "PO8": 0.0, "PO9": 0.7024, "PO10": 0.6745,
        "PO11": 0.0, "PSO2": 0.7076, "PSO3": 0.7205, "PSO4": 0.0,
    }
    expected_po_flags = {
        po: ("Y" if w > 0 else "N") for po, w in expected_po_weighted.items()
    }
    for po in PO_IDS:
        assert float(po_rows[po]["weighted_attainment"]) == pytest.approx(
            expected_po_weighted[po], abs=1e-4
        )
        assert po_rows[po]["target_achieved"] == expected_po_flags[po]

    with paths["course_summary"].open() as fh:
        summary = json.load(fh)
    assert summary["co_count"] == 4
    assert summary["po_count"] == 12
    # avg CO scaled = (2.27 + 2.02 + 2.19 + 2.01) / 4 = 8.49 / 4 = 2.1225 -> 2.12
    assert summary["avg_co_scaled"] == pytest.approx(2.12, abs=1e-9)
    # avg PO scaled = (4*2.12 + 2*2.16 + 2.11 + 2.02 + 4*0) / 12 = 16.93/12 -> 1.41
    assert summary["avg_po_scaled"] == pytest.approx(1.41, abs=1e-9)
    assert summary["co_target_achieved_pct"] == pytest.approx(100.0)
    # 8 of 12 POs achieve the 1.4 target -> 66.67%
    assert summary["po_target_achieved_pct"] == pytest.approx(66.67, abs=1e-9)

    with paths["target_achievement"].open() as fh:
        target_rows = list(csv.DictReader(fh))
    assert len(target_rows) == 16  # 4 COs + 12 POs
    assert all(row["target"] == "1.4" for row in target_rows)
