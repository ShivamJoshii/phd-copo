from __future__ import annotations

import csv
import json
import tempfile
from io import StringIO
from pathlib import Path

import streamlit as st

from copo_mapper.pipeline import run_pairwise_mapping

COLOR_BY_STRENGTH = {
    0: "#f8d7da",
    1: "#fff3cd",
    2: "#d1ecf1",
    3: "#d4edda",
}


def _pick_value(row: dict[str, str], candidates: list[str]) -> str | None:
    lowered = {key.lower(): value for key, value in row.items()}
    for candidate in candidates:
        value = lowered.get(candidate.lower())
        if value is not None:
            return value
    return None


def _load_json(uploaded_file) -> list[dict[str, str]]:
    raw = uploaded_file.getvalue().decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of objects.")
    normalized = []
    for item in data:
        item_id = _pick_value(item, ["id", "co", "po"])
        item_text = _pick_value(item, ["text", "description"])
        if item_id is None or item_text is None:
            raise ValueError(
                "Each item must include an ID field (id/CO/PO) and a text field (text/description)."
            )
        normalized.append({"id": str(item_id).strip(), "text": str(item_text).strip()})
    return normalized


def _load_csv(uploaded_file) -> list[dict[str, str]]:
    raw = uploaded_file.getvalue().decode("utf-8")
    rows = list(csv.DictReader(StringIO(raw)))
    normalized_rows = []
    for item in rows:
        lowered = {key.lower(): value for key, value in item.items()}
        item_id = lowered.get("id") or lowered.get("co") or lowered.get("po")
        item_text = lowered.get("text") or lowered.get("description")
        if item_id is None or item_text is None:
            raise ValueError(
                "CSV must include ID column (id/CO/PO) and text column (text/Description)."
            )
        normalized_rows.append({"id": str(item_id).strip(), "text": str(item_text).strip()})
    return normalized_rows


def _load_outcomes(uploaded_file) -> list[dict[str, str]]:
    filename = uploaded_file.name.lower()
    if filename.endswith(".json"):
        return _load_json(uploaded_file)
    if filename.endswith(".csv"):
        return _load_csv(uploaded_file)
    raise ValueError("Unsupported file format. Please upload .json or .csv files.")


def _load_csv(uploaded_file) -> list[dict[str, str]]:
    raw = uploaded_file.getvalue().decode("utf-8")
    rows = list(csv.DictReader(StringIO(raw)))
    normalized_rows = []
    for item in rows:
        lowered = {key.lower(): value for key, value in item.items()}
        item_id = lowered.get("id") or lowered.get("co") or lowered.get("po")
        item_text = lowered.get("text") or lowered.get("description")
        if item_id is None or item_text is None:
            raise ValueError(
                "CSV must include ID column (id/CO/PO) and text column (text/Description)."
            )
        normalized_rows.append({"id": str(item_id).strip(), "text": str(item_text).strip()})
    return normalized_rows


def _load_outcomes(uploaded_file) -> list[dict[str, str]]:
    filename = uploaded_file.name.lower()
    if filename.endswith(".json"):
        return _load_json(uploaded_file)
    if filename.endswith(".csv"):
        return _load_csv(uploaded_file)
    raise ValueError("Unsupported file format. Please upload .json or .csv files.")


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


def main() -> None:
    st.set_page_config(page_title="CO-PO Mapper UI", layout="wide")
    st.title("CO-PO Mapping Inspector")
    st.write("Upload CO and PO files (.json or .csv), run mapping, inspect matrix and pair-level details.")

    with st.sidebar:
        st.header("Inputs")
        co_upload = st.file_uploader("Upload CO file", type=["json", "csv"])
        po_upload = st.file_uploader("Upload PO file", type=["json", "csv"])

    if co_upload is None or po_upload is None:
        st.info("Please upload both CO and PO files (.json or .csv) to continue.")
        return

    if st.button("Run Mapping", type="primary"):
        try:
            co_data = _load_outcomes(co_upload)
            po_data = _load_outcomes(po_upload)
        except ValueError as err:
            st.error(str(err))
            return

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            co_path = tmp_path / "co.json"
            po_path = tmp_path / "po.json"
            out_dir = tmp_path / "out"

            co_path.write_text(json.dumps(co_data))
            po_path.write_text(json.dumps(po_data))
            pair_path, matrix_path = run_pairwise_mapping(str(co_path), str(po_path), str(out_dir))

            pair_rows = _read_csv_rows(pair_path)
            matrix_header, matrix_rows = _read_matrix(matrix_path)

        st.session_state["pair_rows"] = pair_rows
        st.session_state["matrix_header"] = matrix_header
        st.session_state["matrix_rows"] = matrix_rows

    if "pair_rows" not in st.session_state:
        return

    pair_rows: list[dict[str, str]] = st.session_state["pair_rows"]
    matrix_header: list[str] = st.session_state["matrix_header"]
    matrix_rows: list[list[str]] = st.session_state["matrix_rows"]

    st.subheader("CO-PO Matrix")
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
        (
            row
            for row in pair_rows
            if row["co_id"] == selected_co and row["po_id"] == selected_po
        ),
        None,
    )

    st.subheader("Pair Details")
    if selected is None:
        st.warning("No matching pair found.")
    else:
        st.write(f"**CO ({selected['co_id']}):** {selected['co_text']}")
        st.write(f"**PO ({selected['po_id']}):** {selected['po_text']}")
        st.write(f"**Predicted strength:** {selected['predicted_strength']}")
        st.write(f"**Confidence:** {selected.get('confidence', 'N/A')}")
        st.write(f"**Explanation:** {selected.get('explanation', 'N/A')}")

    pair_buffer = StringIO()
    pair_writer = csv.DictWriter(pair_buffer, fieldnames=list(pair_rows[0].keys()))
    pair_writer.writeheader()
    pair_writer.writerows(pair_rows)
    pair_csv = pair_buffer.getvalue()

    matrix_buffer = StringIO()
    matrix_writer = csv.writer(matrix_buffer)
    matrix_writer.writerow(matrix_header)
    matrix_writer.writerows(matrix_rows)
    matrix_csv = matrix_buffer.getvalue()

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Export Pair Predictions CSV",
            data=pair_csv,
            file_name="pair_predictions.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Export Matrix CSV",
            data=matrix_csv,
            file_name="matrix.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
