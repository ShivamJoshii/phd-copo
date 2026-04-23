from __future__ import annotations

import csv
import json
import tempfile
from io import StringIO
from pathlib import Path

import streamlit as st

from copo_mapper.attainment import run_attainment_analysis
from copo_mapper.pipeline import (
    CO_ID_KEY,
    CO_TEXT_KEY,
    PO_ID_KEY,
    PO_TEXT_KEY,
    run_pairwise_mapping,
)

OUTCOME_UPLOAD_TYPES = ["json", "csv"]
TABULAR_UPLOAD_TYPES = ["json", "csv"]

COLOR_BY_STRENGTH = {
    0: "#f8d7da",
    1: "#fff3cd",
    2: "#d1ecf1",
    3: "#d4edda",
}


def _upload_suffix(uploaded_file) -> str:
    name = (uploaded_file.name or "").lower()
    if name.endswith(".csv"):
        return ".csv"
    if name.endswith(".json"):
        return ".json"
    raise ValueError(f"Unsupported file type: {uploaded_file.name}. Use .json or .csv.")


def _load_outcome_upload(uploaded_file, id_key: str, text_key: str) -> list[dict[str, str]]:
    suffix = _upload_suffix(uploaded_file)
    raw = uploaded_file.getvalue().decode("utf-8")
    if suffix == ".csv":
        rows: list[dict[str, str]] = list(csv.DictReader(StringIO(raw)))
    else:
        rows = json.loads(raw)
        if not isinstance(rows, list):
            raise ValueError(
                f"JSON must be a list of objects with '{id_key}' and '{text_key}' fields."
            )
    for item in rows:
        if id_key not in item or text_key not in item:
            raise ValueError(f"Each row must include '{id_key}' and '{text_key}'.")
    return rows


def _load_generic_upload(uploaded_file) -> list[dict] | dict:
    suffix = _upload_suffix(uploaded_file)
    raw = uploaded_file.getvalue().decode("utf-8")
    if suffix == ".csv":
        return list(csv.DictReader(StringIO(raw)))
    return json.loads(raw)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _read_matrix(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open() as f:
        reader = list(csv.reader(f))
    return reader[0], reader[1:]


def _matrix_html(header: list[str], rows: list[list[str]]) -> str:
    html = [
        "<table style='border-collapse: collapse; width: 100%;'>",
        "<thead><tr>",
    ]
    for col in header:
        html.append(f"<th style='border:1px solid #ccc; padding:6px; background:#f2f2f2'>{col}</th>")
    html.append("</tr></thead><tbody>")

    for row in rows:
        html.append("<tr>")
        for i, val in enumerate(row):
            if i == 0:
                html.append(f"<td style='border:1px solid #ccc; padding:6px; font-weight:600'>{val}</td>")
            else:
                strength = int(val)
                bg = COLOR_BY_STRENGTH.get(strength, "#ffffff")
                html.append(
                    f"<td style='border:1px solid #ccc; padding:6px; text-align:center; background:{bg}'>{val}</td>"
                )
        html.append("</tr>")

    html.append("</tbody></table>")
    return "".join(html)


def _csv_from_rows(rows: list[dict[str, str]]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _csv_from_matrix(header: list[str], rows: list[list[str]]) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def _mapping_tab() -> None:
    st.subheader("Stage 1 — CO-PO Mapping")
    st.write("Upload CO/PO JSON, generate pairwise mapping, and inspect matrix + pair details.")

    with st.sidebar:
        st.header("Stage 1 Inputs")
        co_upload = st.file_uploader(
            f"Upload CO file (columns: {CO_ID_KEY}, {CO_TEXT_KEY})",
            type=OUTCOME_UPLOAD_TYPES,
            key="co_upload",
        )
        po_upload = st.file_uploader(
            f"Upload PO file (columns: {PO_ID_KEY}, {PO_TEXT_KEY})",
            type=OUTCOME_UPLOAD_TYPES,
            key="po_upload",
        )

    if co_upload is None or po_upload is None:
        st.info("Please upload both CO and PO files (JSON or CSV) to run mapping.")
        return

    if st.button("Run Mapping", type="primary"):
        try:
            _load_outcome_upload(co_upload, CO_ID_KEY, CO_TEXT_KEY)
            _load_outcome_upload(po_upload, PO_ID_KEY, PO_TEXT_KEY)
            co_suffix = _upload_suffix(co_upload)
            po_suffix = _upload_suffix(po_upload)
        except ValueError as err:
            st.error(str(err))
            return

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            co_path = tmp_path / f"co{co_suffix}"
            po_path = tmp_path / f"po{po_suffix}"
            out_dir = tmp_path / "out"

            co_path.write_bytes(co_upload.getvalue())
            po_path.write_bytes(po_upload.getvalue())
            pair_path, matrix_path = run_pairwise_mapping(str(co_path), str(po_path), str(out_dir))

            pair_rows = _read_csv_rows(pair_path)
            matrix_header, matrix_rows = _read_matrix(matrix_path)

        st.session_state["pair_rows"] = pair_rows
        st.session_state["matrix_header"] = matrix_header
        st.session_state["matrix_rows"] = matrix_rows
        st.session_state["matrix_csv"] = _csv_from_matrix(matrix_header, matrix_rows)

    if "pair_rows" not in st.session_state:
        return

    pair_rows: list[dict[str, str]] = st.session_state["pair_rows"]
    matrix_header: list[str] = st.session_state["matrix_header"]
    matrix_rows: list[list[str]] = st.session_state["matrix_rows"]

    st.markdown(_matrix_html(matrix_header, matrix_rows), unsafe_allow_html=True)
    st.caption("Color scale: 0=red, 1=yellow, 2=blue, 3=green")

    co_ids = sorted({row["co_id"] for row in pair_rows})
    po_ids = sorted({row["po_id"] for row in pair_rows})

    left, right = st.columns(2)
    with left:
        selected_co = st.selectbox("Select CO", co_ids)
    with right:
        selected_po = st.selectbox("Select PO", po_ids)

    selected = next(
        (row for row in pair_rows if row["co_id"] == selected_co and row["po_id"] == selected_po),
        None,
    )

    st.subheader("Pair Details")
    if selected is not None:
        st.write(f"**CO ({selected['co_id']}):** {selected['co_text']}")
        st.write(f"**PO ({selected['po_id']}):** {selected['po_text']}")
        st.write(f"**Predicted strength:** {selected['predicted_strength']}")
        st.write(f"**Confidence:** {selected.get('confidence', 'N/A')}")
        st.write(f"**Explanation:** {selected.get('explanation', 'N/A')}")

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Export Pair Predictions CSV",
            data=_csv_from_rows(pair_rows),
            file_name="pair_predictions.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Export Matrix CSV",
            data=st.session_state["matrix_csv"],
            file_name="matrix.csv",
            mime="text/csv",
        )


def _attainment_tab() -> None:
    st.subheader("Stage 2 — Attainment Analysis (Connected)")
    st.write("Use mapping matrix from Stage 1 or upload a matrix CSV, then run attainment roll-up.")

    with st.sidebar:
        st.header("Stage 2 Inputs")
        co_att_upload = st.file_uploader(
            "Upload CO Attainment (JSON or CSV)",
            type=TABULAR_UPLOAD_TYPES,
            key="co_att_upload",
            help="Columns: co_id, ma_attainment, ea_attainment, indirect_attainment",
        )
        config_upload = st.file_uploader(
            "Upload Attainment Config (JSON or CSV)",
            type=TABULAR_UPLOAD_TYPES,
            key="config_upload",
            help="Keys: ma_weight, ea_weight, direct_weight, indirect_weight, co_target_level, po_target_level",
        )
        matrix_upload = st.file_uploader(
            "Optional: Upload Mapping Matrix CSV",
            type=["csv"],
            key="matrix_upload",
            help="If omitted, Stage 2 will use the Stage 1 matrix from this app session.",
        )

    if co_att_upload is None or config_upload is None:
        st.info("Upload CO attainment and config files (JSON or CSV) to run Stage 2.")
        return

    if st.button("Run Attainment Analysis", type="primary"):
        try:
            _load_generic_upload(co_att_upload)
            _load_generic_upload(config_upload)
            co_att_suffix = _upload_suffix(co_att_upload)
            cfg_suffix = _upload_suffix(config_upload)
        except ValueError as err:
            st.error(str(err))
            return

        matrix_csv = None
        if matrix_upload is not None:
            matrix_csv = matrix_upload.getvalue().decode("utf-8")
        else:
            matrix_csv = st.session_state.get("matrix_csv")

        if matrix_csv is None:
            st.error("No mapping matrix available. Run Stage 1 first or upload matrix CSV in Stage 2.")
            return

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            co_att_path = tmp_path / f"co_attainment{co_att_suffix}"
            cfg_path = tmp_path / f"attainment_config{cfg_suffix}"
            matrix_path = tmp_path / "matrix.csv"
            out_dir = tmp_path / "attainment_out"

            co_att_path.write_bytes(co_att_upload.getvalue())
            cfg_path.write_bytes(config_upload.getvalue())
            matrix_path.write_text(matrix_csv)

            paths = run_attainment_analysis(
                co_attainment_file=str(co_att_path),
                mapping_matrix_file=str(matrix_path),
                config_file=str(cfg_path),
                out_dir=str(out_dir),
            )

            co_summary = _read_csv_rows(paths["co_summary"])
            po_summary = _read_csv_rows(paths["po_summary"])
            target_summary = _read_csv_rows(paths["target_achievement"])
            course_summary = json.loads(paths["course_summary"].read_text())

        st.session_state["co_summary"] = co_summary
        st.session_state["po_summary"] = po_summary
        st.session_state["target_summary"] = target_summary
        st.session_state["course_summary"] = course_summary

    if "co_summary" not in st.session_state:
        return

    st.markdown("### CO Attainment Summary")
    st.dataframe(st.session_state["co_summary"], use_container_width=True)

    st.markdown("### PO Attainment Summary")
    st.dataframe(st.session_state["po_summary"], use_container_width=True)

    st.markdown("### Target Achievement")
    st.dataframe(st.session_state["target_summary"], use_container_width=True)

    st.markdown("### Course Summary")
    st.json(st.session_state["course_summary"])

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Export CO Summary CSV",
            data=_csv_from_rows(st.session_state["co_summary"]),
            file_name="co_attainment_summary.csv",
            mime="text/csv",
        )
    with d2:
        st.download_button(
            "Export PO Summary CSV",
            data=_csv_from_rows(st.session_state["po_summary"]),
            file_name="po_attainment_summary.csv",
            mime="text/csv",
        )
    with d3:
        st.download_button(
            "Export Target Achievement CSV",
            data=_csv_from_rows(st.session_state["target_summary"]),
            file_name="target_achievement.csv",
            mime="text/csv",
        )


def main() -> None:
    st.set_page_config(page_title="CO-PO Mapper + Attainment UI", layout="wide")
    st.title("CO-PO Mapping & Attainment Workbench")

    tab_map, tab_att = st.tabs(["Stage 1: Mapping", "Stage 2: Attainment"])
    with tab_map:
        _mapping_tab()
    with tab_att:
        _attainment_tab()


if __name__ == "__main__":
    main()
