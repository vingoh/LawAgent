"""
Run citation prediction pipeline on a query CSV and write predictions.

Usage:
    conda run -n agent python src/query/run.py
    conda run -n agent python src/query/run.py --input dataset/test.csv --output results/test_predictions.csv
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from retrieval.bm25 import retrieve_bm25

ROOT_DIR    = os.path.join(os.path.dirname(__file__), "../..")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
DEFAULT_INPUT  = os.path.join(DATASET_DIR, "val.csv")
DEFAULT_OUTPUT = os.path.join(ROOT_DIR, "results/predictions.csv")


def predict_citations(
    query: str,
    k: int = 200,
    k_court: int = 300,
    k_law: int = 300,
    weight_extracted: float = 2.0,
    weight_law: float = 1.2,
    weight_court: float = 1.0,
    rrf_k: int = 60,
) -> list[str]:
    """Baseline pipeline: dual BM25 + query extraction RRF fusion."""
    return retrieve_bm25(
        query,
        k=k,
        k_court=k_court,
        k_law=k_law,
        weight_extracted=weight_extracted,
        weight_law=weight_law,
        weight_court=weight_court,
        rrf_k=rrf_k,
    )


def format_citations(citations: list[str]) -> str:
    return "; ".join(citations)


def load_queries(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    fieldnames = rows[0].keys()
    for col in ("query_id", "query"):
        if col not in fieldnames:
            raise ValueError(f"Input CSV missing required column: {col}")
    return rows


def write_predictions(path: str, rows: list[tuple[str, str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "predicted_citations"])
        writer.writeheader()
        for qid, pred in rows:
            writer.writerow({"query_id": qid, "predicted_citations": pred})


def run(
    input_path: str,
    output_path: str,
    k: int,
    k_court: int,
    k_law: int,
    weight_extracted: float,
    weight_law: float,
    weight_court: float,
    rrf_k: int,
) -> None:
    queries = load_queries(input_path)
    results: list[tuple[str, str]] = []
    for i, row in enumerate(queries, 1):
        qid = row["query_id"]
        print(f"[{i}/{len(queries)}] {qid}", file=sys.stderr)
        citations = predict_citations(
            row["query"],
            k=k,
            k_court=k_court,
            k_law=k_law,
            weight_extracted=weight_extracted,
            weight_law=weight_law,
            weight_court=weight_court,
            rrf_k=rrf_k,
        )
        results.append((qid, format_citations(citations)))
    write_predictions(output_path, results)
    print(f"Wrote {len(results)} predictions → {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run query pipeline and write predictions CSV")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input query CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output predictions CSV")
    parser.add_argument("--k-court", type=int, default=300)
    parser.add_argument("--k-law", type=int, default=300)
    parser.add_argument("--k", type=int, default=200, help="RRF output top-k")
    parser.add_argument("--weight-extracted", type=float, default=2.0)
    parser.add_argument("--weight-law", type=float, default=1.2)
    parser.add_argument("--weight-court", type=float, default=1.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    args = parser.parse_args()
    run(
        args.input,
        args.output,
        k=args.k,
        k_court=args.k_court,
        k_law=args.k_law,
        weight_extracted=args.weight_extracted,
        weight_law=args.weight_law,
        weight_court=args.weight_court,
        rrf_k=args.rrf_k,
    )


if __name__ == "__main__":
    main()
