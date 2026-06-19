"""
Build BM25 indexes from corpus.pkl using bm25s.

Inputs:  indexes/corpus.pkl
Outputs: indexes/bm25_court/  (court documents)
         indexes/bm25_law/    (law documents)
"""

import os
import pickle
import tracemalloc

import bm25s

from bm25_tokenize import tokenize_for_bm25

INDEX_DIR   = os.path.join(os.path.dirname(__file__), "../../indexes")
CORPUS_PATH = os.path.join(INDEX_DIR, "corpus.pkl")


def _build_index(docs: list[dict], output_dir: str, label: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    indexed_texts = [doc["indexed_text"] for doc in docs]
    citations     = [doc["citation"] for doc in docs]
    print(f"Tokenizing {label} ({len(docs):,} docs) ...")
    tracemalloc.start()
    tokenized = tokenize_for_bm25(indexed_texts, citations=citations, show_progress=True)
    retriever = bm25s.BM25()
    retriever.index(tokenized, show_progress=True)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Peak memory during {label} indexing: {peak / 1e9:.2f} GB")
    retriever.save(output_dir, corpus=indexed_texts)
    print(f"{label} index saved → {output_dir}/")


def main() -> None:
    print("Loading corpus.pkl ...")
    with open(CORPUS_PATH, "rb") as f:
        corpus: list[dict] = pickle.load(f)
    print(f"Corpus size: {len(corpus):,} documents")

    court_docs = [d for d in corpus if d["source"] == "court"]
    law_docs   = [d for d in corpus if d["source"] == "law"]
    _build_index(court_docs, os.path.join(INDEX_DIR, "bm25_court"), "court")
    _build_index(law_docs,   os.path.join(INDEX_DIR, "bm25_law"),   "law")


if __name__ == "__main__":
    main()
