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
from retrieval import corpus
from retrieval.bm25 import _load_index as _load_bm25_index, retrieve_bm25_parts
from retrieval.dense import (
    _load_index as _load_dense_index,
    dense_court_exists,
    dense_law_exists,
    retrieve_dense_parts,
)
from retrieval.rerank import _load_reranker, rerank_with_scores, reranker_exists
from retrieval.rrf import weighted_rrf
from retrieval.selector import run_selector

ROOT_DIR    = os.path.join(os.path.dirname(__file__), "../..")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
DEFAULT_INPUT  = os.path.join(DATASET_DIR, "val.csv")
DEFAULT_OUTPUT = os.path.join(ROOT_DIR, "results/predictions.csv")
DEFAULT_REWRITE_LOG_DIR = os.path.join(ROOT_DIR, "results/rewrite_logs")
_USE_DENSE_COURT: bool = dense_court_exists()
_USE_DENSE_LAW:   bool = dense_law_exists()
_USE_RERANK: bool = reranker_exists()


def _print_run_config(
    input_path: str,
    output_path: str,
    *,
    use_rewrite: bool,
    rewrite_log_dir: str | None,
    use_rerank: bool,
    rerank_top_k: int,
    rerank_batch_size: int,
    use_select: bool,
    use_llm_verify: bool,
    verifier_top_k: int,
) -> None:
    print("Run configuration:", file=sys.stderr)
    print(f"  ROOT_DIR                = {ROOT_DIR}", file=sys.stderr)
    print(f"  DATASET_DIR             = {DATASET_DIR}", file=sys.stderr)
    print(f"  DEFAULT_INPUT           = {DEFAULT_INPUT}", file=sys.stderr)
    print(f"  DEFAULT_OUTPUT          = {DEFAULT_OUTPUT}", file=sys.stderr)
    print(f"  DEFAULT_REWRITE_LOG_DIR = {DEFAULT_REWRITE_LOG_DIR}", file=sys.stderr)
    print(f"  dense_court             = {_USE_DENSE_COURT}", file=sys.stderr)
    print(f"  dense_law               = {_USE_DENSE_LAW}", file=sys.stderr)
    print(f"  reranker                = {_USE_RERANK and use_rerank} (top_k={rerank_top_k}, batch={rerank_batch_size})", file=sys.stderr)
    print(f"  selector                = {use_select} (llm_verify={use_llm_verify}, verifier_top_k={verifier_top_k})", file=sys.stderr)
    print(f"  input (actual)          = {input_path}", file=sys.stderr)
    print(f"  output (actual)         = {output_path}", file=sys.stderr)
    print(f"  use_rewrite             = {use_rewrite}", file=sys.stderr)
    print(f"  rewrite_log_dir         = {rewrite_log_dir}", file=sys.stderr)


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
        "expected_articles": result.expected_articles,
        "search_terms": result.search_terms,
        "search_text": search_text,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def predict_citations(
    query: str,
    *,
    use_rewrite: bool = True,
    use_rerank: bool = True,
    rerank_top_k: int = 100,
    rerank_batch_size: int = 32,
    use_select: bool = True,
    use_llm_verify: bool = True,
    verifier_top_k: int = 60,
    query_id: str | None = None,
    rewrite_log_dir: str | None = None,
    k: int = 700,
    k_court: int = 300,
    k_law: int = 300,
    weight_extracted: float = 2.0,
    weight_bm25_court: float = 1.0,
    weight_bm25_law: float = 1.2,
    weight_dense_court: float = 0.6,
    weight_dense_law: float = 1.2,
    rrf_k: int = 60,
    # deprecated aliases kept for backward compatibility
    weight_court: float | None = None,
    weight_law: float | None = None,
) -> list[str]:
    """Query pipeline: optional LLM rewrite + 5-way weighted RRF + optional reranker + selector."""
    if weight_court is not None:
        weight_bm25_court = weight_court
    if weight_law is not None:
        weight_bm25_law = weight_law

    search_text = None
    rewrite_result = None
    llm_articles: list[str] = []
    if use_rewrite:
        rewrite_result = rewrite_query(query)
        search_text = format_search_text(rewrite_result, lang="de")
        llm_articles = rewrite_result.expected_articles
        if llm_articles:
            search_text = search_text + " " + " ".join(llm_articles)
        if rewrite_log_dir and query_id:
            _log_rewrite(rewrite_log_dir, query_id, query, rewrite_result, search_text)

    extracted, bm25_court, bm25_law = retrieve_bm25_parts(
        query, search_text, k_court, k_law,
        extra_citations=llm_articles or None,
    )

    if llm_articles:
        seen: set[str] = set(llm_articles)
        merged = list(llm_articles)
        merged.extend(c for c in extracted if c not in seen)
        extracted = merged

    rankings: list[tuple[list[str], float]] = [
        (extracted,   weight_extracted),
        (bm25_court,  weight_bm25_court),
        (bm25_law,    weight_bm25_law),
    ]

    dense_court: list[str] = []
    dense_law: list[str] = []
    if _USE_DENSE_COURT or _USE_DENSE_LAW:
        dense_query = search_text if search_text is not None else query
        dense_court, dense_law = retrieve_dense_parts(
            dense_query,
            k_court=k_court,
            k_law=k_law,
            use_court=_USE_DENSE_COURT,
            use_law=_USE_DENSE_LAW,
        )
        if _USE_DENSE_COURT and dense_court:
            rankings.append((dense_court, weight_dense_court))
        if _USE_DENSE_LAW and dense_law:
            rankings.append((dense_law, weight_dense_law))

    rrf_result, rrf_scores = weighted_rrf(rankings, rrf_k=rrf_k)

    if use_rerank and _USE_RERANK:
        rerank_query = query
        if search_text is not None:
            rerank_query = query + " " + search_text
        scored = rerank_with_scores(
            rerank_query,
            rrf_result,
            corpus.get_corpus_texts(),
            top_k=rerank_top_k,
            batch_size=rerank_batch_size,
        )
    else:
        # No reranker: use rrf_scores as proxy, convert to scored list
        scored = [(cit, rrf_scores.get(cit, 0.0)) for cit in rrf_result[:rerank_top_k]]

    if use_select:
        source_rankings: dict[str, list[str]] = {
            "extracted": extracted,
            "bm25_court": bm25_court,
            "bm25_law": bm25_law,
        }
        if _USE_DENSE_COURT and dense_court:
            source_rankings["dense_court"] = dense_court
        if _USE_DENSE_LAW and dense_law:
            source_rankings["dense_law"] = dense_law

        return run_selector(
            query,
            rewrite_result,
            scored,
            rrf_scores,
            source_rankings,
            corpus.get_corpus_texts(),
            use_llm_verify=use_llm_verify,
            verifier_top_k=verifier_top_k,
        )

    # Fallback: no selector — return flat reranked list truncated to k
    return [cit for cit, _ in scored][:k]


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
    use_rerank: bool,
    rerank_top_k: int,
    rerank_batch_size: int,
    use_select: bool,
    use_llm_verify: bool,
    verifier_top_k: int,
    rewrite_log_dir: str | None,
    k: int,
    k_court: int,
    k_law: int,
    weight_extracted: float,
    weight_bm25_court: float,
    weight_bm25_law: float,
    weight_dense_court: float,
    weight_dense_law: float,
    rrf_k: int,
) -> tuple[str, str]:
    qid = row["query_id"]
    citations = predict_citations(
        row["query"],
        use_rewrite=use_rewrite,
        use_rerank=use_rerank,
        rerank_top_k=rerank_top_k,
        rerank_batch_size=rerank_batch_size,
        use_select=use_select,
        use_llm_verify=use_llm_verify,
        verifier_top_k=verifier_top_k,
        query_id=qid,
        rewrite_log_dir=rewrite_log_dir,
        k=k,
        k_court=k_court,
        k_law=k_law,
        weight_extracted=weight_extracted,
        weight_bm25_court=weight_bm25_court,
        weight_bm25_law=weight_bm25_law,
        weight_dense_court=weight_dense_court,
        weight_dense_law=weight_dense_law,
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
    weight_bm25_court: float,
    weight_bm25_law: float,
    weight_dense_court: float,
    weight_dense_law: float,
    rrf_k: int,
    use_rewrite: bool = True,
    use_rerank: bool = True,
    rerank_top_k: int = 100,
    rerank_batch_size: int = 32,
    use_select: bool = True,
    use_llm_verify: bool = True,
    verifier_top_k: int = 60,
    rewrite_log_dir: str | None = None,
    workers: int = 4,
) -> None:
    _print_run_config(
        input_path,
        output_path,
        use_rewrite=use_rewrite,
        rewrite_log_dir=rewrite_log_dir,
        use_rerank=use_rerank,
        rerank_top_k=rerank_top_k,
        rerank_batch_size=rerank_batch_size,
        use_select=use_select,
        use_llm_verify=use_llm_verify,
        verifier_top_k=verifier_top_k,
    )
    queries = load_queries(input_path)
    if not queries:
        write_predictions(output_path, [])
        print(f"Wrote 0 predictions → {output_path}", file=sys.stderr)
        return

    corpus.load_corpus()
    _load_bm25_index()
    if _USE_DENSE_COURT or _USE_DENSE_LAW:
        _load_dense_index(use_court=_USE_DENSE_COURT, use_law=_USE_DENSE_LAW)
    if use_rerank and _USE_RERANK:
        _load_reranker()

    process_kwargs = {
        "use_rewrite": use_rewrite,
        "use_rerank": use_rerank,
        "rerank_top_k": rerank_top_k,
        "rerank_batch_size": rerank_batch_size,
        "use_select": use_select,
        "use_llm_verify": use_llm_verify,
        "verifier_top_k": verifier_top_k,
        "rewrite_log_dir": rewrite_log_dir,
        "k": k,
        "k_court": k_court,
        "k_law": k_law,
        "weight_extracted": weight_extracted,
        "weight_bm25_court": weight_bm25_court,
        "weight_bm25_law": weight_bm25_law,
        "weight_dense_court": weight_dense_court,
        "weight_dense_law": weight_dense_law,
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
    parser.add_argument("--input",  default=DEFAULT_INPUT,  help="Input query CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output predictions CSV")
    parser.add_argument("--k-court", type=int, default=300)
    parser.add_argument("--k-law",   type=int, default=300)
    parser.add_argument("--k",       type=int, default=200, help="RRF output top-k")
    parser.add_argument("--weight-extracted",   type=float, default=2.0)
    parser.add_argument("--weight-bm25-court",  type=float, default=1.0)
    parser.add_argument("--weight-bm25-law",    type=float, default=1.2)
    parser.add_argument("--weight-dense-court", type=float, default=0.6)
    parser.add_argument("--weight-dense-law",   type=float, default=1.2)
    # deprecated aliases
    parser.add_argument("--weight-court", type=float, default=None,
                        help="Deprecated alias for --weight-bm25-court")
    parser.add_argument("--weight-law",   type=float, default=None,
                        help="Deprecated alias for --weight-bm25-law")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--no-rewrite", action="store_true", help="Skip LLM query rewrite")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Skip reranker (default: reranker ON if model exists)")
    parser.add_argument("--rerank-top-k", type=int, default=100,
                        help="Number of RRF candidates sent to reranker (default: 100)")
    parser.add_argument("--rerank-batch-size", type=int, default=32,
                        help="Reranker inference batch size (default: 32)")
    parser.add_argument(
        "--no-select",
        action="store_true",
        help="Disable selector pipeline; fall back to plain reranked[:k] output",
    )
    parser.add_argument(
        "--no-llm-verify",
        action="store_true",
        help="Run selector without LLM verifier (cheap expand + score fusion only)",
    )
    parser.add_argument(
        "--verifier-top-k",
        type=int,
        default=60,
        help="Number of candidates shown to LLM verifier (default: 60)",
    )
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

    weight_bm25_court = args.weight_court if args.weight_court is not None else args.weight_bm25_court
    weight_bm25_law   = args.weight_law   if args.weight_law   is not None else args.weight_bm25_law

    run(
        args.input,
        args.output,
        k=args.k,
        k_court=args.k_court,
        k_law=args.k_law,
        weight_extracted=args.weight_extracted,
        weight_bm25_court=weight_bm25_court,
        weight_bm25_law=weight_bm25_law,
        weight_dense_court=args.weight_dense_court,
        weight_dense_law=args.weight_dense_law,
        rrf_k=args.rrf_k,
        use_rewrite=not args.no_rewrite,
        use_rerank=not args.no_rerank,
        rerank_top_k=args.rerank_top_k,
        rerank_batch_size=args.rerank_batch_size,
        use_select=not args.no_select,
        use_llm_verify=not args.no_llm_verify,
        verifier_top_k=args.verifier_top_k,
        rewrite_log_dir=None if args.no_rewrite_log else args.rewrite_log,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
