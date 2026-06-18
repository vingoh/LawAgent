"""
Evaluate BM25 retrieval on val.csv using Macro F1 (exact string match),
matching Kaggle competition scoring.

Usage:
    conda run -n agent python src/eval/macro_f1.py
    conda run -n agent python src/eval/macro_f1.py --k 50 100 200
    conda run -n agent python src/eval/macro_f1.py --k 100 --output results/val_bm25.tsv
"""

import argparse
import csv
import os
import pickle
import re
import sys

import bm25s

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indexing.bm25_tokenize import tokenize_for_bm25

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR    = os.path.join(os.path.dirname(__file__), "../..")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
INDEX_DIR   = os.path.join(ROOT_DIR, "indexes")
BM25_DIR    = os.path.join(INDEX_DIR, "bm25")
CORPUS_PATH = os.path.join(INDEX_DIR, "corpus.pkl")

VAL_CSV  = os.path.join(DATASET_DIR, "val.csv")

# ---------------------------------------------------------------------------
# Citation regex patterns (for extracting citations embedded in query text)
# ---------------------------------------------------------------------------
STATUTE_RE = re.compile(
    r'Art\.\s*\d+(?:\s+Abs\.\s*\d+(?:\s+lit\.\s*\w+)?)?'
    r'(?:\s+[A-Z][A-Za-z]+)+',
)
BGE_RE  = re.compile(r'BGE\s+\d+\s+[IVX]+\s+\d+(?:\s+E\.\s*[\d.]+)?')
BGER_RE = re.compile(r'\d[A-Z]_\d+/\d{4}(?:\s+E\.\s*[\d.a-zA-Z]+)?')


def extract_citations_from_query(query: str) -> list[str]:
    """Extract citation strings literally embedded in the query text."""
    found = []
    for pattern in (BGE_RE, BGER_RE, STATUTE_RE):
        for m in pattern.finditer(query):
            cit = re.sub(r'\s+', ' ', m.group(0)).strip()
            if cit not in found:
                found.append(cit)
    return found


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

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


def compute_recall_at_k(gold: set[str], candidates: list[str], k: int) -> float:
    """Recall@k over an ordered candidate list."""
    if not gold:
        return 0.0
    hits = len(gold & set(candidates[:k]))
    return hits / len(gold)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

_retriever: bm25s.BM25 | None = None
_corpus:    list[dict] | None  = None


def _load_index() -> None:
    global _retriever, _corpus
    if _retriever is not None:
        return
    print("Loading BM25 index ...", file=sys.stderr)
    _retriever = bm25s.BM25.load(BM25_DIR, load_corpus=False)
    print("Loading corpus.pkl ...", file=sys.stderr)
    with open(CORPUS_PATH, "rb") as f:
        _corpus = pickle.load(f)
    print(f"Ready. Corpus size: {len(_corpus):,}", file=sys.stderr)


def retrieve_bm25(query: str, k: int = 200) -> list[str]:
    """Return up to k citation strings from BM25, then prepend any citations
    extracted directly from the query text (exact match, highest priority).
    """
    _load_index()

    query_extracted = extract_citations_from_query(query)
    tokenized_q = tokenize_for_bm25(
        [query], citations=[query_extracted], show_progress=False
    )
    results, _ = _retriever.retrieve(tokenized_q, k=k)
    bm25_indices = results[0].tolist()

    bm25_citations = [_corpus[i]["citation"] for i in bm25_indices]

    # Prepend citations extracted from query text (deduped, order preserved)
    merged = list(query_extracted)
    for c in bm25_citations:
        if c not in merged:
            merged.append(c)

    return merged[:k + len(query_extracted)]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_val(top_k: int = 100, extra_ks: list[int] | None = None) -> dict:
    """Run BM25 retrieval on all val queries. Returns result dict."""
    extra_ks = sorted(set(extra_ks or [50, 100, 200]))
    max_k = max(extra_ks + [top_k])

    rows = []
    with open(VAL_CSV, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    results = []
    for row in rows:
        qid   = row["query_id"]
        query = row["query"]
        gold  = {c.strip() for c in row["gold_citations"].split(";") if c.strip()}

        candidates = retrieve_bm25(query, k=max_k)
        pred_topk  = set(candidates[:top_k])

        p, r, f1 = _query_f1(gold, pred_topk)
        recall_at = {k_: compute_recall_at_k(gold, candidates, k_) for k_ in extra_ks}
        tp = len(gold & pred_topk)
        missed = sorted(gold - pred_topk)

        results.append({
            "query_id":  qid,
            "gold":      len(gold),
            "pred":      len(pred_topk),
            "tp":        tp,
            "precision": p,
            "recall":    r,
            "f1":        f1,
            "recall_at": recall_at,
            "missed":    missed,
        })

    return {"top_k": top_k, "extra_ks": extra_ks, "rows": results}


def print_results(data: dict, output_path: str | None = None) -> None:
    top_k    = data["top_k"]
    extra_ks = data["extra_ks"]
    rows     = data["rows"]

    recall_cols = [f"R@{k}" for k in extra_ks]
    header = (
        f"{'query_id':<12} | {'gold':>4} | {'tp':>5} | "
        + " | ".join(f"{c:>6}" for c in recall_cols)
        + f" | {'F1@'+str(top_k):>8} | missed (first 3)"
    )
    sep = "-" * len(header)

    lines = [header, sep]
    all_f1, all_recall = [], {k: [] for k in extra_ks}

    for r in rows:
        recall_str = " | ".join(f"{r['recall_at'][k]:>6.3f}" for k in extra_ks)
        missed_preview = str(r["missed"][:3])[1:-1] if r["missed"] else "-"
        lines.append(
            f"{r['query_id']:<12} | {r['gold']:>4} | {r['tp']:>5} | "
            f"{recall_str} | {r['f1']:>8.4f} | {missed_preview}"
        )
        all_f1.append(r["f1"])
        for k in extra_ks:
            all_recall[k].append(r["recall_at"][k])

    lines.append(sep)
    avg_recall = " | ".join(f"{sum(all_recall[k])/len(rows):>6.3f}" for k in extra_ks)
    macro_f1   = sum(all_f1) / len(rows)
    lines.append(
        f"{'AGGREGATE':<12} | {'':>4} | {'':>5} | "
        f"{avg_recall} | {macro_f1:>8.4f} |"
    )

    output = "\n".join(lines)
    print(output)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output + "\n")
        print(f"\nResults saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BM25 on val.csv")
    parser.add_argument(
        "--k", type=int, nargs="+", default=[50, 100, 200],
        help="Recall@k values to report. The last value is also used as top-k for F1.",
    )
    parser.add_argument(
        "--f1-k", type=int, default=None,
        help="Top-k for F1 (default: last value in --k).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Optional path to save results as a text file.",
    )
    args = parser.parse_args()

    k_values = sorted(set(args.k))
    top_k    = args.f1_k if args.f1_k else k_values[-1]

    print(f"Evaluating BM25 on val.csv  |  top_k={top_k}  recall_ks={k_values}\n")
    data = evaluate_val(top_k=top_k, extra_ks=k_values)
    print_results(data, output_path=args.output)


if __name__ == "__main__":
    main()
