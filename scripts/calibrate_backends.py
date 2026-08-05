#!/usr/bin/env python3
"""CLI for the empirical SBERT/BERT anchor calibration in copo_mapper.calibration.

Runs each requested backend through the banded hand-written pairs, the
near-paraphrase ceiling probes, and (unless ``--skip-real``) the full real
CO x PO grid from the institutional exports, then prints suggested rescale
anchors and flags drift > 0.05 against scoring.SIMILARITY_RESCALE.

The same measurements are available in the Streamlit app's Calibration tab,
which is the practical way to run this on Streamlit Cloud where the neural
dependencies are installed from requirements.txt.

Requires the neural dependencies:
    pip install sentence-transformers   # sbert
    pip install transformers torch      # bert

Run from the repo root:
    python3 scripts/calibrate_backends.py --backend both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from copo_mapper.calibration import (  # noqa: E402
    DRIFT_TOLERANCE,
    calibrate,
)
from copo_mapper.ingest import parse_real_co_file, parse_real_po_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sbert", "bert", "both"), default="both")
    parser.add_argument("--sbert-model", default=None, help="override sbert model name")
    parser.add_argument("--bert-model", default=None, help="override bert model name")
    parser.add_argument("--co-file", type=Path, default=REPO_ROOT.parent / "CO (1).csv")
    parser.add_argument("--po-file", type=Path, default=REPO_ROOT.parent / "PO (1).csv")
    parser.add_argument("--skip-real", action="store_true", help="skip the real-grid pass")
    parser.add_argument("--batch-size", type=int, default=128, help="pairs per model batch")
    parser.add_argument("--json-out", type=Path, default=None, help="also write the full report as JSON")
    args = parser.parse_args()

    real_co_texts: list[str] | None = None
    real_po_texts: list[str] | None = None
    if not args.skip_real:
        if args.co_file.exists() and args.po_file.exists():
            cos = parse_real_co_file(args.co_file)
            pos = parse_real_po_file(args.po_file)
            real_co_texts = [r.description for r in cos]
            real_po_texts = [p.description for p in pos if p.kind in ("PO", "PSO")]
        else:
            print(f"real exports not found at {args.co_file} / {args.po_file}; skipping real-grid pass")

    backends = ("sbert", "bert") if args.backend == "both" else (args.backend,)
    exit_code = 0
    reports = []

    for backend in backends:
        model = args.sbert_model if backend == "sbert" else args.bert_model
        print("=" * 72)
        print(f"backend={backend} model={model or 'default'}")
        print("=" * 72)

        report = calibrate(
            backend,
            model=model,
            real_co_texts=real_co_texts,
            real_po_texts=real_po_texts,
            batch_size=args.batch_size,
        )
        if report is None:
            print(f"{backend}: dependencies unavailable — install them and re-run.")
            exit_code = max(exit_code, 1)
            continue
        reports.append(report)

        print(f"thresholds={report.thresholds} anchors={report.anchors}")
        print("\n--- banded pairs ---")
        for row in report.banded_rows:
            marker = "ok " if row.predicted == row.expected else "MISS"
            print(
                f"[{marker}] band={row.band:9s} expected={row.expected} predicted={row.predicted} "
                f"raw={row.raw:.3f} rescaled={row.rescaled:.3f} composite={row.composite:.3f}"
            )
        print("\nRaw similarity by band:")
        for s in report.band_summaries:
            print(
                f"  {s.band:9s} n={s.n} min={s.raw_min:.3f} max={s.raw_max:.3f} mean={s.raw_mean:.3f}"
            )
        print(
            f"{len(report.banded_rows) - report.banded_mismatches}/{len(report.banded_rows)} "
            "banded pairs matched the expected label."
        )

        print("\n--- paraphrase ceiling probes ---")
        print("raw sims: " + " ".join(f"{s:.3f}" for s in report.paraphrase_sims))

        if report.real_grid:
            grid = report.real_grid
            print(
                f"\n--- real grid: {grid['n_cos']} COs x {grid['n_pos']} POs/PSOs "
                f"= {grid['n_pairs']} pairs ---"
            )
            print(
                f"raw cosine: min={grid['raw_min']:.3f} max={grid['raw_max']:.3f} "
                f"mean={grid['raw_mean']:.3f}"
            )
            for name, values in grid["percentiles"].items():
                print(f"  {name:<3s} raw={values['raw']:.3f} rescaled={values['rescaled']:.3f}")
            dist = "  ".join(
                f"{lab}:{grid['label_counts'][lab]} ({grid['label_percent'][lab]}%)"
                for lab in ("0", "1", "2", "3")
            )
            print(f"label distribution through score_pair: {dist}")
            print("reference tfidf-validated program distribution: 0:26%  1:72%  2:1.7%  3:0.2%")

        suggestion = report.suggestion
        print(f"\n--- anchor suggestion ({backend}) ---")
        print(f"current   anchors: lo={suggestion.lo_current:.3f} hi={suggestion.hi_current:.3f}")
        print(f"suggested anchors: lo={suggestion.lo_suggested:.3f} hi={suggestion.hi_suggested:.3f}")
        if suggestion.flagged:
            print(
                f"DRIFT EXCEEDS {DRIFT_TOLERANCE}: consider updating "
                f"scoring.SIMILARITY_RESCALE['{backend}'] "
                f"(lo drift {suggestion.lo_drift:.3f}, hi drift {suggestion.hi_drift:.3f}); "
                f"re-check SIMILARITY_FLOOR_FOR_3['{backend}'] = {suggestion.floor3_current:.2f} "
                "against the moderate/strong band boundary as well."
            )
        else:
            print(f"drift within tolerance ({DRIFT_TOLERANCE}); current anchors stand.")
        print()
        if report.banded_mismatches:
            exit_code = max(exit_code, 2)

    if args.json_out and reports:
        args.json_out.write_text(
            json.dumps([r.to_dict() for r in reports], indent=2), encoding="utf-8"
        )
        print(f"wrote JSON report: {args.json_out}")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
