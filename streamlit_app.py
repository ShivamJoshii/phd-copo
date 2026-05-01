from __future__ import annotations

import csv
import json
import tempfile
from io import StringIO
from pathlib import Path

import streamlit as st

from copo_mapper.attainment import (
    COAttainmentInput,
    WeightConfig,
    load_co_attainment_input,
    load_mapping_matrix,
    run_attainment_analysis_from_objects,
)
from copo_mapper.io_utils import normalize_keys
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
    id_target = id_key.strip().lower()
    text_target = text_key.strip().lower()
    for item in rows:
        normalized = normalize_keys(item)
        if id_target not in normalized or text_target not in normalized:
            raise ValueError(
                f"Each row must include '{id_key}' and '{text_key}' (case-insensitive)."
            )
    return rows


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _read_matrix(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open() as f:
        reader = list(csv.reader(f))
    return reader[0], reader[1:]


def _read_matrix_from_string(text: str) -> tuple[list[str], list[list[str]]]:
    reader = list(csv.reader(StringIO(text)))
    return reader[0], reader[1:]


def _co_attainment_template_csv(co_ids: list[str]) -> str:
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["co_id", "ma_attainment", "ea_attainment", "indirect_attainment"])
    for cid in co_ids:
        writer.writerow([cid, 0.0, 0.0, 0.0])
    return buffer.getvalue()


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
        semantic_backend = st.selectbox(
            "Semantic Backend",
            options=["tfidf", "sbert", "bert"],
            index=0,
            help=(
                "Choose similarity engine for Stage 1 mapping. "
                "SBERT/BERT runs require their dependencies and model load to succeed."
            ),
            key="semantic_backend",
        )
        default_model_by_backend = {
            "tfidf": "",
            "sbert": "sentence-transformers/all-MiniLM-L6-v2",
            "bert": "google-bert/bert-base-uncased",
        }
        semantic_model_override = st.text_input(
            "Semantic Model (optional override)",
            value="",
            help=(
                "Model checkpoint name for selected backend. "
                "Leave empty to use backend default."
            ),
            key="semantic_model",
        ).strip()
        semantic_model = semantic_model_override or default_model_by_backend[semantic_backend]

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
            pair_path, matrix_path = run_pairwise_mapping(
                str(co_path),
                str(po_path),
                str(out_dir),
                semantic_backend=semantic_backend,
                semantic_model=semantic_model or None,
            )

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
        st.write(f"**Semantic method used:** {selected.get('semantic_method', 'N/A')}")
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
    st.subheader("Stage 2 — Attainment Analysis")
    st.write(
        "Fill the CO table with MA (Internal) / EA (End-Semester) / Indirect values, pick the "
        "weight splits and target level, then run. The mapping matrix from Stage 1 provides the "
        "CO list and the PO/PSO columns."
    )

    with st.sidebar:
        st.header("Stage 2 Inputs")
        matrix_upload = st.file_uploader(
            "Optional: Upload Mapping Matrix CSV",
            type=["csv"],
            key="matrix_upload",
            help="If omitted, Stage 2 uses the Stage 1 matrix from this app session.",
        )
        prefill_upload = st.file_uploader(
            "Optional: Pre-fill CO Attainment (CSV or JSON)",
            type=TABULAR_UPLOAD_TYPES,
            key="co_att_prefill",
            help="Columns: co_id, ma_attainment, ea_attainment, indirect_attainment. "
            "Values populate the editable table below; you can still edit them.",
        )

    if matrix_upload is not None:
        matrix_csv = matrix_upload.getvalue().decode("utf-8")
    else:
        matrix_csv = st.session_state.get("matrix_csv")

    if matrix_csv is None:
        st.info("No mapping matrix available. Run Stage 1 first, or upload a matrix CSV in the sidebar.")
        return

    _matrix_header, matrix_body = _read_matrix_from_string(matrix_csv)
    co_ids = [row[0] for row in matrix_body]

    table_key = "co_attainment_table"
    version_key = "co_editor_version"

    if st.session_state.get("co_attainment_ids") != co_ids:
        prior = {row["co_id"]: row for row in st.session_state.get(table_key, [])}
        st.session_state[table_key] = [
            prior.get(cid, {"co_id": cid, "MA": 0.0, "EA": 0.0, "Indirect": 0.0})
            for cid in co_ids
        ]
        st.session_state["co_attainment_ids"] = co_ids
        st.session_state[version_key] = st.session_state.get(version_key, 0) + 1

    if prefill_upload is not None and st.session_state.get("_prefill_fid") != prefill_upload.file_id:
        prefill_error: str | None = None
        loaded: list[COAttainmentInput] = []
        try:
            suffix = _upload_suffix(prefill_upload)
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                tf.write(prefill_upload.getvalue())
                tmp_name = tf.name
            try:
                loaded = load_co_attainment_input(tmp_name)
            finally:
                Path(tmp_name).unlink(missing_ok=True)
        except (ValueError, KeyError) as err:
            prefill_error = str(err)

        if prefill_error is not None:
            st.error(f"Prefill failed: {prefill_error}")
        else:
            by_id = {item.co_id: item for item in loaded}
            new_rows = []
            for row in st.session_state[table_key]:
                item = by_id.get(row["co_id"])
                if item is None:
                    new_rows.append(row)
                else:
                    new_rows.append(
                        {
                            "co_id": row["co_id"],
                            "MA": item.ma_attainment,
                            "EA": item.ea_attainment,
                            "Indirect": item.indirect_attainment,
                        }
                    )
            st.session_state[table_key] = new_rows
            st.session_state["_prefill_fid"] = prefill_upload.file_id
            st.session_state[version_key] = st.session_state.get(version_key, 0) + 1
            st.rerun()

    st.markdown("### CO Attainment (edit values per CO)")
    edited = st.data_editor(
        st.session_state[table_key],
        key=f"co_editor_v{st.session_state.get(version_key, 0)}",
        num_rows="fixed",
        use_container_width=True,
        column_config={
            "co_id": st.column_config.TextColumn("CO", disabled=True),
            "MA": st.column_config.NumberColumn(
                "MA (Internal)", min_value=0.0, max_value=1.0, step=0.0001, format="%.4f"
            ),
            "EA": st.column_config.NumberColumn(
                "EA (End-Semester)", min_value=0.0, max_value=1.0, step=0.0001, format="%.4f"
            ),
            "Indirect": st.column_config.NumberColumn(
                "Indirect", min_value=0.0, max_value=1.0, step=0.0001, format="%.4f"
            ),
        },
    )
    st.session_state[table_key] = edited

    st.download_button(
        "Download CO Attainment Template (CSV)",
        data=_co_attainment_template_csv(co_ids),
        file_name="co_attainment_template.csv",
        mime="text/csv",
    )

    st.markdown("### Weights & Target")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        ma_weight = st.number_input(
            "MA weight", min_value=0.0, max_value=1.0, value=0.4, step=0.05, key="ma_weight"
        )
        st.caption(f"EA weight = {1 - ma_weight:.2f}")
    with col_b:
        direct_weight = st.number_input(
            "Direct weight", min_value=0.0, max_value=1.0, value=0.8, step=0.05, key="direct_weight"
        )
        st.caption(f"Indirect weight = {1 - direct_weight:.2f}")
    with col_c:
        target_level = st.number_input(
            "Target level (scaled 0–3)",
            min_value=0.0,
            max_value=3.0,
            value=1.4,
            step=0.1,
            key="target_level",
        )

    if st.button("Run Attainment Analysis", type="primary"):
        try:
            co_inputs = [
                COAttainmentInput(
                    co_id=str(row["co_id"]),
                    ma_attainment=float(row.get("MA") or 0.0),
                    ea_attainment=float(row.get("EA") or 0.0),
                    indirect_attainment=float(row.get("Indirect") or 0.0),
                )
                for row in edited
            ]
        except (TypeError, ValueError) as err:
            st.error(f"Invalid value in CO attainment table: {err}")
            return

        config = WeightConfig(
            ma_weight=float(ma_weight),
            ea_weight=float(1 - ma_weight),
            direct_weight=float(direct_weight),
            indirect_weight=float(1 - direct_weight),
            co_target_level=float(target_level),
            po_target_level=float(target_level),
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            matrix_path = tmp_path / "matrix.csv"
            matrix_path.write_text(matrix_csv)
            mapping = load_mapping_matrix(str(matrix_path))

            out_dir = tmp_path / "attainment_out"
            paths = run_attainment_analysis_from_objects(
                co_inputs, mapping, config, str(out_dir)
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
