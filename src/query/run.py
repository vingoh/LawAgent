"""
Run citation prediction pipeline on a query CSV and write predictions.

Usage:
    conda run -n agent python src/query/run.py
    conda run -n agent python src/query/run.py --input dataset/test.csv --output results/test_predictions.csv
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from query.rewrite import format_search_text, rewrite_query
from retrieval.bm25 import _load_index, retrieve_bm25

ROOT_DIR    = os.path.join(os.path.dirname(__file__), "../..")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
DEFAULT_INPUT  = os.path.join(DATASET_DIR, "val.csv")
DEFAULT_OUTPUT = os.path.join(ROOT_DIR, "results/predictions.csv")
DEFAULT_REWRITE_LOG_DIR = os.path.join(ROOT_DIR, "results/rewrite_logs")


def _log_rewrite(
    log_dir: str,
    query_id: str,
    query: str,
    result,
    search_text: str,
) -> None:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{query_id}.json")
    payload = {
        "query_id": query_id,
        "query": query,
        "legal_issue": result.legal_issue,
        "expected_codes": result.expected_codes,
        "search_terms": result.search_terms,
        "search_text": search_text,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def predict_citations(
    query: str,
    *,
    use_rewrite: bool = True,
    query_id: str | None = None,
    rewrite_log_dir: str | None = None,
    k: int = 700,
    k_court: int = 300,
    k_law: int = 300,
    weight_extracted: float = 2.0,
    weight_law: float = 1.2,
    weight_court: float = 1.0,
    rrf_k: int = 60,
) -> list[str]:
    """Query pipeline: optional LLM rewrite + dual BM25 + extraction RRF."""
    search_text = None
    if use_rewrite:
        rewrite_result = rewrite_query(query)
        search_text = format_search_text(rewrite_result, lang="de")
        if rewrite_log_dir and query_id:
            _log_rewrite(rewrite_log_dir, query_id, query, rewrite_result, search_text)
    return retrieve_bm25(
        query,
        search_text=search_text,
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


def _process_query(
    row: dict[str, str],
    *,
    use_rewrite: bool,
    rewrite_log_dir: str | None,
    k: int,
    k_court: int,
    k_law: int,
    weight_extracted: float,
    weight_law: float,
    weight_court: float,
    rrf_k: int,
) -> tuple[str, str]:
    qid = row["query_id"]
    citations = predict_citations(
        row["query"],
        use_rewrite=use_rewrite,
        query_id=qid,
        rewrite_log_dir=rewrite_log_dir,
        k=k,
        k_court=k_court,
        k_law=k_law,
        weight_extracted=weight_extracted,
        weight_law=weight_law,
        weight_court=weight_court,
        rrf_k=rrf_k,
    )
    return qid, format_citations(citations)


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
    use_rewrite: bool = True,
    rewrite_log_dir: str | None = None,
    workers: int = 4,
) -> None:
    queries = load_queries(input_path)
    if not queries:
        write_predictions(output_path, [])
        print(f"Wrote 0 predictions → {output_path}", file=sys.stderr)
        return

    _load_index()
    process_kwargs = {
        "use_rewrite": use_rewrite,
        "rewrite_log_dir": rewrite_log_dir,
        "k": k,
        "k_court": k_court,
        "k_law": k_law,
        "weight_extracted": weight_extracted,
        "weight_law": weight_law,
        "weight_court": weight_court,
        "rrf_k": rrf_k,
    }

    results: dict[int, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_query, row, **process_kwargs): i
            for i, row in enumerate(queries)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            qid, _ = results[i]
            print(f"[{done}/{len(queries)}] {qid}", file=sys.stderr)

    write_predictions(
        output_path, [results[i] for i in range(len(queries))]
    )
    print(f"Wrote {len(results)} predictions → {output_path}", file=sys.stderr)
    if use_rewrite and rewrite_log_dir:
        print(f"Rewrite logs → {rewrite_log_dir}", file=sys.stderr)


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
    parser.add_argument("--no-rewrite", action="store_true", help="Skip LLM query rewrite")
    parser.add_argument(
        "--rewrite-log",
        default=DEFAULT_REWRITE_LOG_DIR,
        help=f"Directory for per-query rewrite JSON logs (default: {DEFAULT_REWRITE_LOG_DIR})",
    )
    parser.add_argument(
        "--no-rewrite-log",
        action="store_true",
        help="Do not write rewrite JSON logs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel worker threads (default: 4; use 1 for sequential)",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
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
        use_rewrite=not args.no_rewrite,
        rewrite_log_dir=None if args.no_rewrite_log else args.rewrite_log,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
