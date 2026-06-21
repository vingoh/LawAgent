"""BM25 retrieval over the law corpus."""

import os
import re
import sys

import bm25s

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indexing.bm25_tokenize import tokenize_for_bm25
from retrieval import corpus as _corpus_mod

ROOT_DIR    = os.path.join(os.path.dirname(__file__), "../..")
INDEX_DIR   = os.path.join(ROOT_DIR, "indexes")
BM25_COURT_DIR = os.path.join(INDEX_DIR, "bm25_court")
BM25_LAW_DIR   = os.path.join(INDEX_DIR, "bm25_law")

_NUM = r'\d+(?:bis|ter|quater|[a-z])?'   # 共用组件：数字 + 可选后缀
STATUTE_RE = re.compile(
    r'Art\.\s*' + _NUM + r'(?:\s+Abs\.\s*' + _NUM + r'(?:\s+lit\.\s*\w+)?)?'
    r'(?:\s+[A-Z][A-Za-z]+)+'
)
BGE_RE  = re.compile(r'BGE\s+\d+\s+[IVX]+\s+\d+(?:\s+E\.\s*[\d.a-zA-Z]+)?')
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
            "Run: conda run -n agent python src/indexing/build_bm25.py"
        )
    print("Loading BM25 court index ...", file=sys.stderr)
    _retriever_court = bm25s.BM25.load(BM25_COURT_DIR, load_corpus=False)
    print("Loading BM25 law index ...", file=sys.stderr)
    _retriever_law = bm25s.BM25.load(BM25_LAW_DIR, load_corpus=False)
    _corpus_court = _corpus_mod.get_corpus_court()
    _corpus_law   = _corpus_mod.get_corpus_law()
    print(
        f"BM25 ready. Court: {len(_corpus_court):,}, law: {len(_corpus_law):,}",
        file=sys.stderr,
    )


def retrieve_bm25_parts(
    query: str,
    search_text: str | None = None,
    k_court: int = 300,
    k_law: int = 300,
    extra_citations: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return (extracted, court_citations, law_citations) without RRF fusion.

    extracted        — citations literally present in the query text
    court_citations  — BM25 court results, ranked by BM25 score
    law_citations    — BM25 law results, ranked by BM25 score
    extra_citations  — additional citations (e.g. LLM-predicted) to inject as
                       atomic citation tokens into the query, without affecting
                       the returned ``extracted`` list
    """
    _load_index()

    extracted = extract_citations_from_query(query)
    text_for_search = search_text if search_text is not None else query

    # Merge extra_citations (e.g. LLM predictions) with query-literal citations
    # so they are all converted to atomic citation tokens by tokenize_for_bm25,
    # enabling exact citation-token matching against the indexed corpus.
    extracted_set = set(extracted)
    all_citations = extracted + [c for c in (extra_citations or []) if c not in extracted_set]

    tokenized_q = tokenize_for_bm25(
        [text_for_search], citations=[all_citations], show_progress=False
    )

    court_results, _ = _retriever_court.retrieve(
        tokenized_q, k=min(k_court, len(_corpus_court))
    )
    law_results, _ = _retriever_law.retrieve(
        tokenized_q, k=min(k_law, len(_corpus_law))
    )

    court_citations = [_corpus_court[i]["citation"] for i in court_results[0].tolist()]
    law_citations   = [_corpus_law[i]["citation"]   for i in law_results[0].tolist()]

    return extracted, court_citations, law_citations
