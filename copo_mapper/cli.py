from __future__ import annotations

import argparse

from .semantic import DEFAULT_SBERT_MODEL
from .pipeline import run_pairwise_mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pairwise CO-PO mapping baseline")
    parser.add_argument("--co-file", required=True, help="Path to CO input file (.json or .csv)")
    parser.add_argument("--po-file", required=True, help="Path to PO input file (.json or .csv)")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument(
        "--semantic-mode",
        choices=["auto", "tfidf", "sbert"],
        default="auto",
        help="Semantic similarity mode: auto (blend with SBERT when available), tfidf, or sbert-only.",
    )
    parser.add_argument(
        "--sbert-model",
        default=DEFAULT_SBERT_MODEL,
        help="SentenceBERT model name/path (used for auto/sbert modes).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pair_path, matrix_path = run_pairwise_mapping(
        args.co_file,
        args.po_file,
        args.out_dir,
        semantic_mode=args.semantic_mode,
        sbert_model=args.sbert_model,
    )
    print(f"Saved pair predictions: {pair_path}")
    print(f"Saved matrix: {matrix_path}")


if __name__ == "__main__":
    main()
