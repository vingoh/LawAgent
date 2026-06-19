"""BM25 retrieval over the law corpus."""

import os
import pickle
import re
import sys

import bm25s

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indexing.bm25_tokenize import tokenize_for_bm25
from retrieval.rrf import weighted_rrf

ROOT_DIR    = os.path.join(os.path.dirname(__file__), "../..")
INDEX_DIR   = os.path.join(ROOT_DIR, "indexes")
BM25_COURT_DIR = os.path.join(INDEX_DIR, "bm25_court")
BM25_LAW_DIR   = os.path.join(INDEX_DIR, "bm25_law")
CORPUS_PATH = os.path.join(INDEX_DIR, "corpus.pkl")

STATUTE_RE = re.compile(
    r'Art\.\s*\d+(?:\s+Abs\.\s*\d+(?:\s+lit\.\s*\w+)?)?'
    r'(?:\s+[A-Z][A-Za-z]+)+',
)
BGE_RE  = re.compile(r'BGE\s+\d+\s+[IVX]+\s+\d+(?:\s+E\.\s*[\d.]+)?')
BGER_RE = re.compile(r'\d[A-Z]_\d+/\d{4}(?:\s+E\.\s*[\d.a-zA-Z]+)?')

_retriever_court: bm25s.BM25 | None = None
_retriever_law:   bm25s.BM25 | None = None
_corpus_court:    list[dict] | None  = None
_corpus_law:      list[dict] | None  = None


def extract_citations_from_query(query: str) -> list[str]:
    """Extract citation strings literally embedded in the query text."""
    found = []
    for pattern in (BGE_RE, BGER_RE, STATUTE_RE):
        for m in pattern.finditer(query):
            cit = re.sub(r'\s+', ' ', m.group(0)).strip()
            if cit not in found:
                found.append(cit)
    return found


def _load_index() -> None:
    global _retriever_court, _retriever_law, _corpus_court, _corpus_law
    if _retriever_court is not None:
        return
    if not os.path.isdir(BM25_COURT_DIR) or not os.path.isdir(BM25_LAW_DIR):
        raise FileNotFoundError(
            f"BM25 indexes not found at {BM25_COURT_DIR} and/or {BM25_LAW_DIR}. "
            "Run: python src/indexing/build_bm25.py"
        )
    print("Loading BM25 court index ...", file=sys.stderr)
    _retriever_court = bm25s.BM25.load(BM25_COURT_DIR, load_corpus=False)
    print("Loading BM25 law index ...", file=sys.stderr)
    _retriever_law = bm25s.BM25.load(BM25_LAW_DIR, load_corpus=False)
    print("Loading corpus.pkl ...", file=sys.stderr)
    with open(CORPUS_PATH, "rb") as f:
        corpus = pickle.load(f)
    _corpus_court = [d for d in corpus if d["source"] == "court"]
    _corpus_law = [d for d in corpus if d["source"] == "law"]
    print(
        f"Ready. Court: {len(_corpus_court):,}, law: {len(_corpus_law):,}",
        file=sys.stderr,
    )


def retrieve_bm25(
    query: str,
    search_text: str | None = None,
    k: int = 700,
    k_court: int = 300,
    k_law: int = 300,
    weight_extracted: float = 2.0,
    weight_law: float = 1.2,
    weight_court: float = 1.0,
    rrf_k: int = 60,
) -> list[str]:
    """Return up to k citation strings via dual BM25 + query extraction RRF fusion."""
    _load_index()

    extracted = extract_citations_from_query(query)
    text_for_search = search_text if search_text is not None else query
    tokenized_q = tokenize_for_bm25(
        [text_for_search], citations=[extracted], show_progress=False
    )

    court_results, _ = _retriever_court.retrieve(
        tokenized_q, k=min(k_court, len(_corpus_court))
    )
    law_results, _ = _retriever_law.retrieve(
        tokenized_q, k=min(k_law, len(_corpus_law))
    )

    court_citations = [_corpus_court[i]["citation"] for i in court_results[0].tolist()]
    law_citations = [_corpus_law[i]["citation"] for i in law_results[0].tolist()]

    rankings: list[tuple[list[str], float]] = []
    if extracted:
        rankings.append((extracted, weight_extracted))
    if court_citations:
        rankings.append((court_citations, weight_court))
    if law_citations:
        rankings.append((law_citations, weight_law))

    fused = weighted_rrf(rankings, rrf_k=rrf_k)
    return fused[:k]
