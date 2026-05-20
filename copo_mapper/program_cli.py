from __future__ import annotations

import argparse

from .aggregate import run_program_aggregation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 4: program-level PO attainment")
    parser.add_argument(
        "--semesters-file",
        required=True,
        help="CSV/JSON with columns: semester_id, credits, PO1, PO2, ...",
    )
    parser.add_argument("--out-dir", default="program_outputs", help="Output directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_path = run_program_aggregation(args.semesters_file, args.out_dir)
    print(f"Saved program PO attainment: {out_path}")


if __name__ == "__main__":
    main()
