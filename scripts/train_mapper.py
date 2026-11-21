"""Fine-tune DistilBERT (or any BERT-family encoder) as a CO-PO mapper.

Run this on a GPU machine or Google Colab; it is NOT used by the app at run
time and needs packages the app does not:

    pip install transformers torch scikit-learn
    python scripts/train_mapper.py --labels labelled_pairs.csv --out mapper_model

Input: one CSV in long format, one row per CO-PO pair from faculty-approved
matrices (the Mapping_Labels + CO/PO master sheets of the data template):

    course_id, co_id, co_text, po_id, po_text, level      (level = 0/1/2/3)

What it does
1. Splits BY COURSE (never by pair), so no CO text is in both train and test.
2. Baseline: the repo's TF-IDF rule scorer on the same test courses.
3. Fine-tunes a sequence-pair classifier ("CO text [SEP] PO text" -> 4 levels)
   with class weights, because most pairs are 0.
4. Reports accuracy, macro-F1, quadratic weighted kappa and the confusion
   matrix for both, and writes them to <out>/metrics.json.

Aim for 1,000+ labelled pairs. With fewer, expect the baseline to win and
report that honestly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from copo_mapper.mapping_eval import evaluate, group_split, load_labelled_pairs  # noqa: E402
from copo_mapper.preprocess import normalize_text  # noqa: E402
from copo_mapper.scoring import score_pair  # noqa: E402
from copo_mapper.semantic import tfidf_pair_similarity  # noqa: E402


def baseline_predictions(rows: list[dict]) -> list[int]:
    cos = [normalize_text(r["co_text"]) for r in rows]
    pos = [normalize_text(r["po_text"]) for r in rows]
    sims = tfidf_pair_similarity(cos, pos)
    return [score_pair(c, p, s, backend="tfidf").score for c, p, s in zip(cos, pos, sims)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="Long-format labelled pairs CSV")
    ap.add_argument("--out", default="mapper_model", help="Output directory")
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline-only", action="store_true", help="Skip fine-tuning")
    args = ap.parse_args()

    rows = load_labelled_pairs(Path(args.labels).read_text(encoding="utf-8-sig"))
    train, test = group_split(rows, args.test_fraction, args.seed)
    if not train or not test:
        sys.exit("Need labelled pairs from at least two courses to split by course.")
    print(f"{len(rows)} pairs, {len({r['course_id'] for r in rows})} courses "
          f"-> train {len(train)}, test {len(test)}")
    print("Train label counts:", dict(sorted(Counter(r["level"] for r in train).items())))

    y_test = [r["level"] for r in test]
    metrics = {"baseline_tfidf_rules": evaluate(y_test, baseline_predictions(test))}
    print("Baseline (TF-IDF + rules):", json.dumps(metrics["baseline_tfidf_rules"]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.baseline_only:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(args.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=4).to(device)

        def collate(batch):
            enc = tok([b["co_text"] for b in batch], [b["po_text"] for b in batch],
                      truncation=True, max_length=256, padding=True, return_tensors="pt")
            enc["labels"] = torch.tensor([b["level"] for b in batch])
            return enc

        counts = Counter(r["level"] for r in train)
        weights = torch.tensor(
            [len(train) / (4 * counts[k]) if counts.get(k) else 0.0 for k in range(4)],
            dtype=torch.float, device=device,
        )
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

        for epoch in range(args.epochs):
            model.train()
            total = 0.0
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                loss = loss_fn(model(**batch).logits, labels)
                loss.backward()
                optim.step()
                optim.zero_grad()
                total += loss.item()
            print(f"epoch {epoch + 1}/{args.epochs} loss {total / len(loader):.4f}")

        model.eval()
        preds: list[int] = []
        with torch.no_grad():
            for batch in DataLoader(test, batch_size=args.batch_size, collate_fn=collate):
                batch = {k: v.to(device) for k, v in batch.items()}
                batch.pop("labels")
                preds += model(**batch).logits.argmax(-1).tolist()
        metrics["fine_tuned"] = {"model": args.model, **evaluate(y_test, preds)}
        print("Fine-tuned:", json.dumps(metrics["fine_tuned"]))
        model.save_pretrained(out)
        tok.save_pretrained(out)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics to {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
