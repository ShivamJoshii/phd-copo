from __future__ import annotations

import csv
import json
import tempfile
from io import StringIO
from pathlib import Path

import streamlit as st

from copo_mapper.aggregate import (
    CreditRow,
    compute_program_po,
    compute_semester_po,
    load_credit_rows,
)
from copo_mapper.attainment import (
    COAttainmentInput,
    WeightConfig,
    compute_co_attainment,
    compute_po_attainment,
    load_co_attainment_input,
    load_mapping_matrix,
    run_attainment_analysis_from_objects,
)
from copo_mapper.diagnostics import diagnose_course
from copo_mapper.ml_drivers import (
    observation_from_co,
    rank_drivers,
    summarize_drivers,
)
from copo_mapper.io_utils import decode_text_bytes, normalize_keys
from copo_mapper.pipeline import (
    CO_ID_KEY,
    CO_TEXT_KEY,
    PO_ID_KEY,
    PO_TEXT_KEY,
    run_pairwise_mapping,
)
from copo_mapper.ui_helpers import (
    DEFAULT_PO_KINDS,
    available_po_kinds,
    canonical_co_csv_text,
    canonical_po_csv_text,
    course_from_option,
    course_options,
    parse_raw_co_bytes,
    parse_raw_po_bytes,
)
from copo_mapper.ingest import to_canonical_co_rows, to_canonical_po_rows

OUTCOME_UPLOAD_TYPES = ["json", "csv"]
FORMAT_RAW = "Raw faculty export"
CO_FORMAT_OPTIONS = [f"Canonical ({CO_ID_KEY},{CO_TEXT_KEY})", FORMAT_RAW]
PO_FORMAT_OPTIONS = [f"Canonical ({PO_ID_KEY},{PO_TEXT_KEY})", FORMAT_RAW]
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
    raw = decode_text_bytes(uploaded_file.getvalue(), source=uploaded_file.name or "upload")
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
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _read_matrix(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(encoding="utf-8-sig") as f:
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


def _raw_co_rows_ui(co_upload) -> list[dict[str, str]] | None:
    """Parse a raw CO export upload, let the user pick a course, and preview.

    Returns canonical ``CO,description`` rows, or None when parsing/conversion
    failed (an st.error has already been shown).
    """
    st.markdown("### Raw CO export → canonical rows")
    try:
        records = parse_raw_co_bytes(co_upload.getvalue())
    except ValueError as err:
        st.error(f"Could not parse raw CO export '{co_upload.name}': {err}")
        return None

    selected_option = st.selectbox(
        "Course to map",
        options=course_options(records),
        key="raw_co_course",
        help=(
            "Pick a single course to map its COs with plain ids (CO1, CO2, ...), "
            "or keep all courses with course-prefixed ids (KMBN101-CO1, ...)."
        ),
    )
    try:
        rows = to_canonical_co_rows(records, course=course_from_option(selected_option))
    except ValueError as err:
        st.error(f"Could not convert CO export: {err}")
        return None

    st.caption(f"{len(rows)} CO row(s) ready for mapping.")
    st.dataframe(rows, width="stretch")
    st.download_button(
        "Download canonical CO CSV",
        data=canonical_co_csv_text(rows),
        file_name="co_canonical.csv",
        mime="text/csv",
    )
    return rows


def _raw_po_rows_ui(po_upload) -> list[dict[str, str]] | None:
    """Parse a raw PO export upload, let the user pick kinds, and preview.

    Returns canonical ``PO,description`` rows, or None when parsing/conversion
    failed (an st.error has already been shown).
    """
    st.markdown("### Raw PO export → canonical rows")
    try:
        records = parse_raw_po_bytes(po_upload.getvalue())
    except ValueError as err:
        st.error(f"Could not parse raw PO export '{po_upload.name}': {err}")
        return None

    kinds = available_po_kinds(records)
    selected_kinds = st.multiselect(
        "Outcome kinds to include",
        options=kinds,
        default=[k for k in kinds if k in DEFAULT_PO_KINDS],
        key="raw_po_kinds",
        help="PEO statements are program educational objectives; excluded by default.",
    )
    if not selected_kinds:
        st.error("Select at least one outcome kind (PO / PSO / PEO) to include.")
        return None
    rows = to_canonical_po_rows(records, include=selected_kinds)

    st.caption(f"{len(rows)} PO row(s) ready for mapping.")
    st.dataframe(rows, width="stretch")
    st.download_button(
        "Download canonical PO CSV",
        data=canonical_po_csv_text(rows),
        file_name="po_canonical.csv",
        mime="text/csv",
    )
    return rows


def _mapping_tab() -> None:
    st.subheader("Stage 1 — CO-PO Mapping")
    st.write(
        "Upload CO/PO files (canonical, or raw faculty exports converted in-app), "
        "generate pairwise mapping, and inspect matrix + pair details."
    )

    with st.sidebar:
        st.header("Stage 1 Inputs")
        co_format = st.radio(
            "CO file format",
            options=CO_FORMAT_OPTIONS,
            key="co_format",
            help=(
                "Canonical: a clean file with CO id and description columns. "
                "Raw faculty export: a single-column institutional dump with course "
                "header rows and lines like 'CO1: ...'."
            ),
        )
        co_upload = st.file_uploader(
            f"Upload CO file (columns: {CO_ID_KEY}, {CO_TEXT_KEY})"
            if co_format != FORMAT_RAW
            else "Upload raw CO export (CSV)",
            type=OUTCOME_UPLOAD_TYPES,
            key="co_upload",
        )
        po_format = st.radio(
            "PO file format",
            options=PO_FORMAT_OPTIONS,
            key="po_format",
            help=(
                "Canonical: a clean file with PO id and description columns. "
                "Raw faculty export: an institutional dump with 'PEO1:', 'PO1:', "
                "'PSO1:' statement lines."
            ),
        )
        po_upload = st.file_uploader(
            f"Upload PO file (columns: {PO_ID_KEY}, {PO_TEXT_KEY})"
            if po_format != FORMAT_RAW
            else "Upload raw PO/PSO/PEO export (CSV)",
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

    co_raw_rows: list[dict[str, str]] | None = None
    po_raw_rows: list[dict[str, str]] | None = None
    if co_format == FORMAT_RAW:
        co_raw_rows = _raw_co_rows_ui(co_upload)
        if co_raw_rows is None:
            return
    if po_format == FORMAT_RAW:
        po_raw_rows = _raw_po_rows_ui(po_upload)
        if po_raw_rows is None:
            return

    if st.button("Run Mapping", type="primary"):
        try:
            if co_raw_rows is not None:
                co_bytes = canonical_co_csv_text(co_raw_rows).encode("utf-8")
                co_suffix = ".csv"
            else:
                _load_outcome_upload(co_upload, CO_ID_KEY, CO_TEXT_KEY)
                co_bytes = co_upload.getvalue()
                co_suffix = _upload_suffix(co_upload)
            if po_raw_rows is not None:
                po_bytes = canonical_po_csv_text(po_raw_rows).encode("utf-8")
                po_suffix = ".csv"
            else:
                _load_outcome_upload(po_upload, PO_ID_KEY, PO_TEXT_KEY)
                po_bytes = po_upload.getvalue()
                po_suffix = _upload_suffix(po_upload)
        except ValueError as err:
            st.error(str(err))
            return

        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                co_path = tmp_path / f"co{co_suffix}"
                po_path = tmp_path / f"po{po_suffix}"
                out_dir = tmp_path / "out"

                co_path.write_bytes(co_bytes)
                po_path.write_bytes(po_bytes)
                pair_path, matrix_path = run_pairwise_mapping(
                    str(co_path),
                    str(po_path),
                    str(out_dir),
                    semantic_backend=semantic_backend,
                    semantic_model=semantic_model or None,
                )

                pair_rows = _read_csv_rows(pair_path)
                matrix_header, matrix_rows = _read_matrix(matrix_path)
        except RuntimeError as err:
            st.error(f"Mapping failed for backend '{semantic_backend}': {err}")
            st.info(
                "Install required dependencies (`pip install -e '.[nlp]'`) and retry. "
                "No fallback is used."
            )
            return

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


def _render_diagnosis(diagnosis) -> None:
    """Render the deterministic root-cause analysis for missed CO/PO targets."""
    st.markdown("### Why did targets miss? (root-cause diagnosis)")
    missed_cos = diagnosis.missed_cos
    missed_pos = diagnosis.missed_pos

    if not missed_cos and not missed_pos:
        st.success("All COs and POs met their target levels — nothing to diagnose.")
        return

    st.caption(
        "Deterministic decomposition of each miss: the weakest input / the CO that "
        "drags the PO down most, plus the cheapest single change that reaches target."
    )

    if missed_pos:
        st.markdown("**Missed POs**")
        for exp in missed_pos:
            with st.expander(f"❌ {exp.po_id} — gap {exp.gap:.2f} (scaled {exp.scaled:.2f} / target {exp.target:.2f})"):
                st.write(exp.reason)
                st.dataframe(
                    [
                        {
                            "CO": c.co_id,
                            "mapping": c.map_strength,
                            "weight %": round(c.weight_share * 100, 1),
                            "CO final": c.co_final,
                            "drag (scaled)": c.drag_scaled,
                        }
                        for c in exp.contributions
                    ],
                    width="stretch",
                )
                if exp.levers:
                    st.caption(
                        "Cheapest fixes (raise a CO's final attainment to this value): "
                        + ", ".join(f"{lv.name}→{lv.needed:.2f}" for lv in exp.levers[:3])
                    )

    if missed_cos:
        st.markdown("**Missed COs**")
        for exp in missed_cos:
            with st.expander(f"❌ {exp.co_id} — gap {exp.gap:.2f} (scaled {exp.scaled:.2f} / target {exp.target:.2f})"):
                st.write(exp.reason)
                st.dataframe(
                    [{"component": k, "value": v} for k, v in exp.components.items()],
                    width="stretch",
                )
                if exp.levers:
                    st.caption(
                        "Single-input fixes: "
                        + ", ".join(f"{lv.name} {lv.current:.2f}→{lv.needed:.2f}" for lv in exp.levers)
                    )


def _render_systemic_drivers() -> None:
    """Cross-course driver analysis built up from each course you run."""
    store = st.session_state.get("co_observations_by_course", {})
    all_obs = [o for rows in store.values() for o in rows]
    n_courses = len(store)

    st.markdown("### Systemic drivers across analysed courses (experimental)")
    st.caption(
        "Builds a dataset from every course you run here, then ranks which input "
        "(internal MA, external EA, indirect) most tracks with missed CO targets. "
        "This is the interpretable, small-data ML layer — directional, not a verdict."
    )

    if n_courses < 2 or len(all_obs) < 6:
        st.info(
            f"Analysed {n_courses} course(s), {len(all_obs)} CO rows so far. "
            "Run at least 2 courses (≥6 CO rows) to unlock driver analysis. "
            "Tip: set distinct Course IDs below before each run so they don't overwrite."
        )
        return

    scores = rank_drivers(all_obs)
    for line in summarize_drivers(scores, len(all_obs)):
        st.write(f"- {line}")
    st.dataframe(
        [
            {
                "feature": s.feature,
                "corr. with miss": s.correlation,
                "mean (met)": s.mean_when_met,
                "mean (missed)": s.mean_when_missed,
                "direction": s.direction,
            }
            for s in scores
        ],
        width="stretch",
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
        matrix_csv = decode_text_bytes(matrix_upload.getvalue(), source=matrix_upload.name or "matrix upload")
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
        width="stretch",
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

        # Deterministic root-cause diagnosis for any missed target.
        co_results = compute_co_attainment(co_inputs, config)
        po_results = compute_po_attainment(co_results, mapping, config)
        diagnosis = diagnose_course(co_results, po_results, mapping, config)

        # Accumulate CO observations across analysed courses for systemic
        # (cross-course) driver analysis. Keyed by course id so re-running a
        # course replaces its earlier rows instead of double-counting.
        course_label = str(st.session_state.get("stage2_course_id") or "course").strip() or "course"
        store = dict(st.session_state.get("co_observations_by_course", {}))
        store[course_label] = [
            observation_from_co(r, config, course_id=course_label) for r in co_results
        ]
        st.session_state["co_observations_by_course"] = store

        st.session_state["co_summary"] = co_summary
        st.session_state["po_summary"] = po_summary
        st.session_state["target_summary"] = target_summary
        st.session_state["course_summary"] = course_summary
        st.session_state["diagnosis"] = diagnosis

    if "co_summary" not in st.session_state:
        return

    st.markdown("### CO Attainment Summary")
    st.dataframe(st.session_state["co_summary"], width="stretch")

    st.markdown("### PO Attainment Summary")
    st.dataframe(st.session_state["po_summary"], width="stretch")

    st.markdown("### Target Achievement")
    st.dataframe(st.session_state["target_summary"], width="stretch")

    st.markdown("### Course Summary")
    st.json(st.session_state["course_summary"])

    if "diagnosis" in st.session_state:
        _render_diagnosis(st.session_state["diagnosis"])

    _render_systemic_drivers()

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

    st.markdown("### Push this course forward to Step 3 (Semester)")
    st.caption(
        "Enter a course id and credits, then add to the semester roll-up. "
        "Course PO values flow through on the 0–3 scale."
    )
    p1, p2, p3 = st.columns([2, 1, 1])
    with p1:
        course_id_input = st.text_input("Course ID", value="Course1", key="stage2_course_id")
    with p2:
        credits_input = st.number_input(
            "Credits", min_value=0.0, value=3.0, step=0.5, key="stage2_course_credits"
        )
    with p3:
        st.write("")
        if st.button("Add course to Step 3"):
            po_row = {
                row["po_id"]: float(row["weighted_attainment"])
                for row in st.session_state["po_summary"]
            }
            new_row: dict = {
                "course_id": course_id_input.strip() or "Course",
                "credits": float(credits_input),
                **po_row,
            }
            existing = [
                r for r in st.session_state.get("semester_courses", [])
                if r.get("course_id") != new_row["course_id"]
            ]
            existing.append(new_row)
            st.session_state["semester_courses"] = existing
            st.session_state["semester_courses_version"] = (
                st.session_state.get("semester_courses_version", 0) + 1
            )
            st.success(f"Added '{new_row['course_id']}' (credits {new_row['credits']}) to Step 3.")


def _rows_to_credit_rows(rows: list[dict]) -> list[CreditRow]:
    credit_rows: list[CreditRow] = []
    for row in rows:
        po_values: dict[str, float] = {}
        id_value = ""
        credits_value = 0.0
        for key, value in row.items():
            if key is None:
                continue
            lower = str(key).strip().lower()
            if lower in {"course_id", "semester_id", "id"}:
                id_value = str(value).strip()
            elif lower in {"credits", "credit"}:
                if value not in (None, ""):
                    credits_value = float(value)
            else:
                if value in (None, ""):
                    continue
                try:
                    po_values[str(key).strip()] = float(value)
                except (TypeError, ValueError):
                    continue
        if not id_value:
            continue
        credit_rows.append(CreditRow(id=id_value, credits=credits_value, po_values=po_values))
    return credit_rows


def _aggregate_results_to_rows(values: dict[str, float], level: str) -> list[dict]:
    return [
        {
            "level": level,
            "po_id": po_id,
            "value": round(value, 4),
            "percentage_on_3_scale": round(value * 100 / 3, 2),
        }
        for po_id, value in values.items()
    ]


def _semester_tab() -> None:
    st.subheader("Step 3 — Semester PO Attainment")
    st.write(
        "Roll up course-level PO values into a semester score, credit-weighted. "
        "Courses pushed forward from Step 2 appear here automatically; you can also "
        "edit rows or upload a `course_id, credits, PO1, PO2, ...` CSV."
    )

    with st.sidebar:
        st.header("Step 3 Inputs")
        courses_upload = st.file_uploader(
            "Optional: Upload courses CSV / JSON",
            type=TABULAR_UPLOAD_TYPES,
            key="courses_upload",
            help="Columns: course_id, credits, PO1, PO2, ...",
        )

    if courses_upload is not None and st.session_state.get("_courses_fid") != courses_upload.file_id:
        try:
            suffix = _upload_suffix(courses_upload)
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                tf.write(courses_upload.getvalue())
                tmp_name = tf.name
            try:
                rows = load_credit_rows(tmp_name)
            finally:
                Path(tmp_name).unlink(missing_ok=True)
        except (ValueError, KeyError) as err:
            st.error(f"Upload failed: {err}")
        else:
            st.session_state["semester_courses"] = [
                {"course_id": r.id, "credits": r.credits, **r.po_values} for r in rows
            ]
            st.session_state["_courses_fid"] = courses_upload.file_id
            st.session_state["semester_courses_version"] = (
                st.session_state.get("semester_courses_version", 0) + 1
            )

    courses = st.session_state.get("semester_courses", [])
    if not courses:
        st.info(
            "No courses yet. Run Step 2 and click 'Add course to Step 3', "
            "or upload a courses CSV in the sidebar."
        )
        return

    st.markdown("### Courses in this semester")
    edited = st.data_editor(
        courses,
        key=f"semester_editor_v{st.session_state.get('semester_courses_version', 0)}",
        num_rows="dynamic",
        width="stretch",
    )
    st.session_state["semester_courses"] = edited

    if st.button("Run Semester Aggregation", type="primary"):
        credit_rows = _rows_to_credit_rows(edited)
        if not credit_rows:
            st.error("Need at least one course row with an id and credits.")
            return
        po_values = compute_semester_po(credit_rows)
        st.session_state["semester_po_values"] = po_values
        st.session_state["semester_summary_rows"] = _aggregate_results_to_rows(po_values, "semester")

    if "semester_summary_rows" not in st.session_state:
        return

    st.markdown("### Semester PO Attainment")
    st.dataframe(st.session_state["semester_summary_rows"], width="stretch")
    st.download_button(
        "Export Semester PO CSV",
        data=_csv_from_rows(st.session_state["semester_summary_rows"]),
        file_name="semester_po_attainment.csv",
        mime="text/csv",
    )

    st.markdown("### Push this semester forward to Step 4 (Program)")
    p1, p2, p3 = st.columns([2, 1, 1])
    with p1:
        sem_id = st.text_input("Semester ID", value="Sem1", key="stage3_semester_id")
    with p2:
        sem_credits = st.number_input(
            "Semester credits", min_value=0.0, value=20.0, step=1.0, key="stage3_semester_credits"
        )
    with p3:
        st.write("")
        if st.button("Add semester to Step 4"):
            new_row: dict = {
                "semester_id": sem_id.strip() or "Sem",
                "credits": float(sem_credits),
                **{k: round(v, 4) for k, v in st.session_state["semester_po_values"].items()},
            }
            existing = [
                r for r in st.session_state.get("program_semesters", [])
                if r.get("semester_id") != new_row["semester_id"]
            ]
            existing.append(new_row)
            st.session_state["program_semesters"] = existing
            st.session_state["program_semesters_version"] = (
                st.session_state.get("program_semesters_version", 0) + 1
            )
            st.success(f"Added '{new_row['semester_id']}' to Step 4.")


def _program_tab() -> None:
    st.subheader("Step 4 — Program / Degree PO Attainment")
    st.write(
        "Aggregate semester-level PO values across the program, credit-weighted. "
        "Semesters pushed forward from Step 3 appear here automatically; or upload a "
        "`semester_id, credits, PO1, PO2, ...` CSV."
    )

    with st.sidebar:
        st.header("Step 4 Inputs")
        semesters_upload = st.file_uploader(
            "Optional: Upload semesters CSV / JSON",
            type=TABULAR_UPLOAD_TYPES,
            key="semesters_upload",
            help="Columns: semester_id, credits, PO1, PO2, ...",
        )

    if semesters_upload is not None and st.session_state.get("_semesters_fid") != semesters_upload.file_id:
        try:
            suffix = _upload_suffix(semesters_upload)
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
                tf.write(semesters_upload.getvalue())
                tmp_name = tf.name
            try:
                rows = load_credit_rows(tmp_name)
            finally:
                Path(tmp_name).unlink(missing_ok=True)
        except (ValueError, KeyError) as err:
            st.error(f"Upload failed: {err}")
        else:
            st.session_state["program_semesters"] = [
                {"semester_id": r.id, "credits": r.credits, **r.po_values} for r in rows
            ]
            st.session_state["_semesters_fid"] = semesters_upload.file_id
            st.session_state["program_semesters_version"] = (
                st.session_state.get("program_semesters_version", 0) + 1
            )

    semesters = st.session_state.get("program_semesters", [])
    if not semesters:
        st.info(
            "No semesters yet. Run Step 3 and click 'Add semester to Step 4', "
            "or upload a semesters CSV in the sidebar."
        )
        return

    st.markdown("### Semesters in this program")
    edited = st.data_editor(
        semesters,
        key=f"program_editor_v{st.session_state.get('program_semesters_version', 0)}",
        num_rows="dynamic",
        width="stretch",
    )
    st.session_state["program_semesters"] = edited

    if st.button("Run Program Aggregation", type="primary"):
        credit_rows = _rows_to_credit_rows(edited)
        if not credit_rows:
            st.error("Need at least one semester row with an id and credits.")
            return
        po_values = compute_program_po(credit_rows)
        st.session_state["program_summary_rows"] = _aggregate_results_to_rows(po_values, "program")

    if "program_summary_rows" not in st.session_state:
        return

    st.markdown("### Program PO Attainment")
    st.dataframe(st.session_state["program_summary_rows"], width="stretch")
    st.download_button(
        "Export Program PO CSV",
        data=_csv_from_rows(st.session_state["program_summary_rows"]),
        file_name="program_po_attainment.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="CO-PO Mapper + Attainment UI", layout="wide")
    st.title("CO-PO Mapping & Attainment Workbench")
    st.caption(
        "Walk left → right: Mapping → Course attainment → Semester roll-up → Program roll-up. "
        "Each step pushes its outputs into the next."
    )

    tab_map, tab_att, tab_sem, tab_prog = st.tabs(
        [
            "Step 1: Mapping",
            "Step 2: Course Attainment",
            "Step 3: Semester",
            "Step 4: Program",
        ]
    )
    with tab_map:
        _mapping_tab()
    with tab_att:
        _attainment_tab()
    with tab_sem:
        _semester_tab()
    with tab_prog:
        _program_tab()


if __name__ == "__main__":
    main()
