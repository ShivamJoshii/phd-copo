from __future__ import annotations

import argparse
from pathlib import Path

from .ingest import (
    list_courses,
    parse_real_co_file,
    parse_real_po_file,
    to_canonical_co_rows,
    to_canonical_po_rows,
    write_canonical_co_csv,
    write_canonical_po_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert messy real-world CO/PO exports into canonical pipeline CSVs"
    )
    parser.add_argument("--co-file", default=None, help="Path to raw CO export CSV")
    parser.add_argument("--po-file", default=None, help="Path to raw PO export CSV")
    parser.add_argument("--out-dir", default="out_ingest", help="Output directory")
    parser.add_argument(
        "--course",
        default=None,
        help="Optional course code filter for COs (e.g. KMBN101). "
        "Without it, CO ids are prefixed with the course code.",
    )
    parser.add_argument(
        "--include",
        default="PO,PSO",
        help="Comma-separated PO kinds to keep (subset of PEO,PO,PSO). Default: PO,PSO.",
    )
    parser.add_argument(
        "--list-courses",
        action="store_true",
        help="Only list the course codes detected in the CO file, then exit.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.co_file is None and args.po_file is None:
        build_parser().error("Provide at least one of --co-file / --po-file.")

    out_dir = Path(args.out_dir)

    if args.co_file is not None:
        co_records = parse_real_co_file(args.co_file)
        if args.list_courses:
            for code, name in list_courses(co_records):
                print(f"{code}: {name}")
            return
        co_rows = to_canonical_co_rows(co_records, course=args.course)
        co_path = write_canonical_co_csv(co_rows, out_dir / "co_canonical.csv")
        print(f"Saved {len(co_rows)} CO rows ({len(list_courses(co_records))} courses parsed): {co_path}")

    if args.po_file is not None:
        include = tuple(part.strip() for part in args.include.split(",") if part.strip())
        po_records = parse_real_po_file(args.po_file)
        po_rows = to_canonical_po_rows(po_records, include=include)
        po_path = write_canonical_po_csv(po_rows, out_dir / "po_canonical.csv")
        print(f"Saved {len(po_rows)} PO rows (kinds: {', '.join(include)}): {po_path}")


if __name__ == "__main__":
    main()
