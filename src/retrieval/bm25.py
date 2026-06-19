"""BM25 retrieval over the law corpus."""

import os
import pickle
import re
import sys

import bm25s

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from indexing.bm25_tokenize import tokenize_for_bm25

ROOT_DIR    = os.path.join(os.path.dirname(__file__), "../..")
INDEX_DIR   = os.path.join(ROOT_DIR, "indexes")
BM25_DIR    = os.path.join(INDEX_DIR, "bm25")
CORPUS_PATH = os.path.join(INDEX_DIR, "corpus.pkl")

STATUTE_RE = re.compile(
    r'Art\.\s*\d+(?:\s+Abs\.\s*\d+(?:\s+lit\.\s*\w+)?)?'
    r'(?:\s+[A-Z][A-Za-z]+)+',
)
BGE_RE  = re.compile(r'BGE\s+\d+\s+[IVX]+\s+\d+(?:\s+E\.\s*[\d.]+)?')
BGER_RE = re.compile(r'\d[A-Z]_\d+/\d{4}(?:\s+E\.\s*[\d.a-zA-Z]+)?')

_retriever: bm25s.BM25 | None = None
_corpus:    list[dict] | None  = None


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
    """Return up to k citation strings from BM25, prepending query-extracted citations."""
    _load_index()

    query_extracted = extract_citations_from_query(query)
    tokenized_q = tokenize_for_bm25(
        [query], citations=[query_extracted], show_progress=False
    )
    results, _ = _retriever.retrieve(tokenized_q, k=k)
    bm25_indices = results[0].tolist()

    bm25_citations = [_corpus[i]["citation"] for i in bm25_indices]

    merged = list(query_extracted)
    for c in bm25_citations:
        if c not in merged:
            merged.append(c)

    return merged[:k + len(query_extracted)]
