from __future__ import annotations

import argparse

from .pipeline import run_pairwise_mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pairwise CO-PO mapping baseline")
    parser.add_argument("--co-file", required=True, help="Path to CO JSON file")
    parser.add_argument("--po-file", required=True, help="Path to PO JSON file")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument(
        "--semantic-backend",
        choices=["tfidf", "sbert", "bert"],
        default="tfidf",
        help="Semantic model backend to use. SBERT/BERT must be available when selected.",
    )
    parser.add_argument(
        "--semantic-model",
        default=None,
        help="Optional model name for selected backend (SBERT/BERT).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pair_path, matrix_path = run_pairwise_mapping(
        args.co_file,
        args.po_file,
        args.out_dir,
        semantic_backend=args.semantic_backend,
        semantic_model=args.semantic_model,
    )
    print(f"Saved pair predictions: {pair_path}")
    print(f"Saved matrix: {matrix_path}")


if __name__ == "__main__":
    main()
