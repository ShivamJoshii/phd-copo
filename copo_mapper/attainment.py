import csv
import json
from dataclasses import dataclass
from pathlib import Path

from .io_utils import normalize_keys


@dataclass(frozen=True)
class WeightConfig:
    ma_weight: float
    ea_weight: float
    direct_weight: float
    indirect_weight: float
    co_target_level: float
    po_target_level: float


@dataclass(frozen=True)
class COAttainmentInput:
    co_id: str
    ma_attainment: float
    ea_attainment: float
    indirect_attainment: float


@dataclass(frozen=True)
class COAttainmentResult:
    co_id: str
    ma_attainment: float
    ea_attainment: float
    indirect_attainment: float
    direct_attainment: float
    final_attainment: float
    scaled_attainment: float
    target_achieved: str


@dataclass(frozen=True)
class POAttainmentResult:
    po_id: str
    weighted_attainment: float
    percentage: float
    scaled_attainment: float
    target_achieved: str


def compute_direct_attainment(ma: float, ea: float, config: WeightConfig) -> float:
    return (ma * config.ma_weight) + (ea * config.ea_weight)


def compute_final_attainment(direct: float, indirect: float, config: WeightConfig) -> float:
    return (direct * config.direct_weight) + (indirect * config.indirect_weight)


def compute_co_attainment(inputs: list[COAttainmentInput], config: WeightConfig) -> list[COAttainmentResult]:
    results: list[COAttainmentResult] = []
    for item in inputs:
        direct = compute_direct_attainment(item.ma_attainment, item.ea_attainment, config)
        final = compute_final_attainment(direct, item.indirect_attainment, config)
        scaled = final * 3
        results.append(
            COAttainmentResult(
                co_id=item.co_id,
                ma_attainment=item.ma_attainment,
                ea_attainment=item.ea_attainment,
                indirect_attainment=item.indirect_attainment,
                direct_attainment=round(direct, 4),
                final_attainment=round(final, 4),
                scaled_attainment=round(scaled, 2),
                target_achieved="Y" if scaled >= config.co_target_level else "N",
            )
        )
    return results


def compute_po_attainment(
    co_results: list[COAttainmentResult],
    mapping: dict[str, dict[str, int]],
    config: WeightConfig,
) -> list[POAttainmentResult]:
    if not mapping:
        return []

    co_lookup = {row.co_id: row.final_attainment for row in co_results}
    po_ids = list(next(iter(mapping.values())).keys())

    results: list[POAttainmentResult] = []
    for po_id in po_ids:
        numerator = 0.0
        denominator = 0.0
        for co_id, po_map in mapping.items():
            map_value = po_map.get(po_id, 0)
            numerator += co_lookup.get(co_id, 0.0) * map_value
            denominator += map_value

        weighted = (numerator / denominator) if denominator else 0.0
        scaled = weighted * 3
        results.append(
            POAttainmentResult(
                po_id=po_id,
                weighted_attainment=round(weighted, 4),
                percentage=round(weighted * 100, 2),
                scaled_attainment=round(scaled, 2),
                target_achieved="Y" if scaled >= config.po_target_level else "N",
            )
        )

    return results


def _read_tabular(path: Path) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    if suffix == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, list):
            raise ValueError(f"{path.name}: JSON input must be a list of objects.")
        return data
    raise ValueError(f"{path.name}: unsupported extension '{suffix}'. Use .json or .csv.")


def load_weight_config(path: str) -> WeightConfig:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        with p.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 1:
            raise ValueError(f"{p.name}: config CSV must contain exactly one data row.")
        data = normalize_keys(rows[0])
    elif suffix == ".json":
        data = normalize_keys(json.loads(p.read_text()))
    else:
        raise ValueError(f"{p.name}: unsupported extension '{suffix}'. Use .json or .csv.")

    return WeightConfig(
        ma_weight=float(data["ma_weight"]),
        ea_weight=float(data["ea_weight"]),
        direct_weight=float(data["direct_weight"]),
        indirect_weight=float(data["indirect_weight"]),
        co_target_level=float(data["co_target_level"]),
        po_target_level=float(data["po_target_level"]),
    )


def load_co_attainment_input(path: str) -> list[COAttainmentInput]:
    results: list[COAttainmentInput] = []
    for row in _read_tabular(Path(path)):
        r = normalize_keys(row)
        results.append(
            COAttainmentInput(
                co_id=str(r["co_id"]).strip(),
                ma_attainment=float(r["ma_attainment"]),
                ea_attainment=float(r["ea_attainment"]),
                indirect_attainment=float(r["indirect_attainment"]),
            )
        )
    return results


def load_mapping_matrix(path: str) -> dict[str, dict[str, int]]:
    p = Path(path)
    with p.open() as f:
        rows = list(csv.DictReader(f))

    mapping: dict[str, dict[str, int]] = {}
    for row in rows:
        co_id_value: str | None = None
        po_cells: dict[str, int] = {}
        for k, v in row.items():
            if k is None:
                continue
            if k.strip().lower() == "co_id":
                co_id_value = v
            elif v not in ("", None):
                po_cells[k] = int(v)
        if co_id_value is None:
            raise ValueError(f"{p.name}: matrix CSV must include a 'co_id' column.")
        mapping[str(co_id_value).strip()] = po_cells
    return mapping


def summarize_course(
    co_results: list[COAttainmentResult],
    po_results: list[POAttainmentResult],
) -> dict[str, float | int]:
    co_count = len(co_results)
    po_count = len(po_results)

    co_achieved = sum(1 for row in co_results if row.target_achieved == "Y")
    po_achieved = sum(1 for row in po_results if row.target_achieved == "Y")

    return {
        "co_count": co_count,
        "po_count": po_count,
        "avg_co_scaled": round(
            (sum(row.scaled_attainment for row in co_results) / co_count) if co_count else 0.0,
            2,
        ),
        "avg_po_scaled": round(
            (sum(row.scaled_attainment for row in po_results) / po_count) if po_count else 0.0,
            2,
        ),
        "co_target_achieved_pct": round((co_achieved / co_count * 100) if co_count else 0.0, 2),
        "po_target_achieved_pct": round((po_achieved / po_count * 100) if po_count else 0.0, 2),
    }


def _write_attainment_outputs(
    co_results: list[COAttainmentResult],
    po_results: list[POAttainmentResult],
    config: WeightConfig,
    out_dir: str,
) -> dict[str, Path]:
    course_summary = summarize_course(co_results, po_results)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    co_path = output_dir / "co_attainment_summary.csv"
    po_path = output_dir / "po_attainment_summary.csv"
    target_path = output_dir / "target_achievement.csv"
    summary_path = output_dir / "course_summary.json"

    with co_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(COAttainmentResult.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([row.__dict__ for row in co_results])

    with po_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(POAttainmentResult.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows([row.__dict__ for row in po_results])

    with target_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "id", "scaled_attainment", "target", "achieved"])
        writer.writeheader()
        for row in co_results:
            writer.writerow(
                {
                    "type": "CO",
                    "id": row.co_id,
                    "scaled_attainment": row.scaled_attainment,
                    "target": config.co_target_level,
                    "achieved": row.target_achieved,
                }
            )
        for row in po_results:
            writer.writerow(
                {
                    "type": "PO",
                    "id": row.po_id,
                    "scaled_attainment": row.scaled_attainment,
                    "target": config.po_target_level,
                    "achieved": row.target_achieved,
                }
            )

    summary_path.write_text(json.dumps(course_summary, indent=2))

    return {
        "co_summary": co_path,
        "po_summary": po_path,
        "target_achievement": target_path,
        "course_summary": summary_path,
    }


def run_attainment_analysis_from_objects(
    co_inputs: list[COAttainmentInput],
    mapping: dict[str, dict[str, int]],
    config: WeightConfig,
    out_dir: str,
) -> dict[str, Path]:
    co_results = compute_co_attainment(co_inputs, config)
    po_results = compute_po_attainment(co_results, mapping, config)
    return _write_attainment_outputs(co_results, po_results, config, out_dir)


def run_attainment_analysis(
    co_attainment_file: str,
    mapping_matrix_file: str,
    config_file: str,
    out_dir: str,
) -> dict[str, Path]:
    config = load_weight_config(config_file)
    co_inputs = load_co_attainment_input(co_attainment_file)
    mapping = load_mapping_matrix(mapping_matrix_file)
    return run_attainment_analysis_from_objects(co_inputs, mapping, config, out_dir)
