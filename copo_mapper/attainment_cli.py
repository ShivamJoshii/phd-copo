from __future__ import annotations

import argparse

from .attainment import run_attainment_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage 2 attainment analysis")
    parser.add_argument("--co-attainment-file", required=True, help="Path to CO attainment JSON")
    parser.add_argument("--mapping-matrix-file", required=True, help="Path to mapping matrix CSV")
    parser.add_argument("--config-file", required=True, help="Path to attainment config JSON")
    parser.add_argument("--out-dir", default="attainment_outputs", help="Output directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = run_attainment_analysis(
        co_attainment_file=args.co_attainment_file,
        mapping_matrix_file=args.mapping_matrix_file,
        config_file=args.config_file,
        out_dir=args.out_dir,
    )
    print(f"Saved CO summary: {paths['co_summary']}")
    print(f"Saved PO summary: {paths['po_summary']}")
    print(f"Saved target achievement: {paths['target_achievement']}")
    print(f"Saved course summary: {paths['course_summary']}")


if __name__ == "__main__":
    main()
