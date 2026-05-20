from __future__ import annotations

import argparse

from .aggregate import run_semester_aggregation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage 3: semester-level PO attainment")
    parser.add_argument(
        "--courses-file",
        required=True,
        help="CSV/JSON with columns: course_id, credits, PO1, PO2, ...",
    )
    parser.add_argument("--out-dir", default="semester_outputs", help="Output directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_path = run_semester_aggregation(args.courses_file, args.out_dir)
    print(f"Saved semester PO attainment: {out_path}")


if __name__ == "__main__":
    main()
