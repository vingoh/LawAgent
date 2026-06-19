"""
Evaluate predictions CSV against val.csv gold using Macro F1 (exact string match).

Usage:
    conda run -n agent python src/eval/macro_f1.py
    conda run -n agent python src/eval/macro_f1.py --predictions results/predictions.csv
    conda run -n agent python src/eval/macro_f1.py --output results/eval_report.txt
"""

import argparse
import csv
import os
import sys

ROOT_DIR    = os.path.join(os.path.dirname(__file__), "../..")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
VAL_CSV = os.path.join(DATASET_DIR, "val.csv")
DEFAULT_PREDICTIONS = os.path.join(ROOT_DIR, "results/predictions.csv")


def parse_citations(s: str) -> set[str]:
    """Parse semicolon-separated citation string into a set."""
    return {c.strip() for c in s.split(";") if c.strip()}


def _query_f1(gold: set[str], pred: set[str]) -> tuple[float, float, float]:
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(gold & pred)
    p  = tp / len(pred)
    r  = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def compute_macro_f1(gold_list: list[set[str]], pred_list: list[set[str]]) -> float:
    """Macro F1: per-query F1 averaged. Exact string match."""
    scores = [_query_f1(g, p)[2] for g, p in zip(gold_list, pred_list)]
    return sum(scores) / len(scores) if scores else 0.0


def load_gold(path: str = VAL_CSV) -> list[tuple[str, set[str]]]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        (row["query_id"], parse_citations(row["gold_citations"]))
        for row in rows
    ]


def load_predictions(path: str) -> dict[str, set[str]]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for col in ("query_id", "predicted_citations"):
        if rows and col not in rows[0]:
            raise ValueError(f"Predictions CSV missing required column: {col}")
    pred_map = {row["query_id"]: parse_citations(row["predicted_citations"]) for row in rows}
    return pred_map


def evaluate_predictions(predictions_path: str, gold_path: str = VAL_CSV) -> dict:
    gold_rows = load_gold(gold_path)
    pred_map = load_predictions(predictions_path)

    extra_ids = set(pred_map) - {qid for qid, _ in gold_rows}
    if extra_ids:
        print(f"Warning: ignoring {len(extra_ids)} query_id(s) not in gold: {sorted(extra_ids)}", file=sys.stderr)

    missing = [qid for qid, _ in gold_rows if qid not in pred_map]
    if missing:
        raise ValueError(f"Predictions missing {len(missing)} query_id(s) from gold: {missing}")

    results = []
    for qid, gold in gold_rows:
        pred = pred_map[qid]
        p, r, f1 = _query_f1(gold, pred)
        tp = len(gold & pred)
        missed = sorted(gold - pred)
        results.append({
            "query_id": qid,
            "gold": len(gold),
            "pred": len(pred),
            "tp": tp,
            "precision": p,
            "recall": r,
            "f1": f1,
            "missed": missed,
        })

    return {"rows": results}


def print_results(data: dict, output_path: str | None = None) -> None:
    rows = data["rows"]
    header = (
        f"{'query_id':<12} | {'gold':>4} | {'pred':>4} | {'tp':>4} | "
        f"{'precision':>9} | {'recall':>6} | {'f1':>8} | missed (first 3)"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for r in rows:
        missed_preview = str(r["missed"][:3])[1:-1] if r["missed"] else "-"
        lines.append(
            f"{r['query_id']:<12} | {r['gold']:>4} | {r['pred']:>4} | {r['tp']:>4} | "
            f"{r['precision']:>9.4f} | {r['recall']:>6.4f} | {r['f1']:>8.4f} | {missed_preview}"
        )

    lines.append(sep)
    macro_f1 = sum(r["f1"] for r in rows) / len(rows) if rows else 0.0
    lines.append(f"{'AGGREGATE':<12} | {'':>4} | {'':>4} | {'':>4} | {'':>9} | {'':>6} | {macro_f1:>8.4f} |")

    output = "\n".join(lines)
    print(output)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output + "\n")
        print(f"\nResults saved → {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions CSV against val.csv gold")
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS, help="Predictions CSV path")
    parser.add_argument("--output", default=None, help="Optional path to save text report")
    args = parser.parse_args()

    print(f"Evaluating {args.predictions} against {VAL_CSV}\n")
    data = evaluate_predictions(args.predictions)
    print_results(data, output_path=args.output)


if __name__ == "__main__":
    main()
