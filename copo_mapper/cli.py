from __future__ import annotations

import argparse

from .pipeline import run_pairwise_mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run pairwise CO-PO mapping baseline")
    parser.add_argument("--co-file", required=True, help="Path to CO JSON file")
    parser.add_argument("--po-file", required=True, help="Path to PO JSON file")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument(
        "--use-sbert",
        action="store_true",
        help="Use SBERT semantic similarity if sentence-transformers is installed.",
    )
    parser.add_argument(
        "--sbert-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-Transformers model name to use with --use-sbert.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pair_path, matrix_path = run_pairwise_mapping(
        args.co_file,
        args.po_file,
        args.out_dir,
        use_sbert=args.use_sbert,
        sbert_model=args.sbert_model,
    )
    print(f"Saved pair predictions: {pair_path}")
    print(f"Saved matrix: {matrix_path}")


if __name__ == "__main__":
    main()
