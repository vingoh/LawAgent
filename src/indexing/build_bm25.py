"""
Build BM25 index from corpus.pkl using bm25s.

Inputs:  indexes/corpus.pkl
Outputs: indexes/bm25/  (bm25s saved index directory)
"""

import os
import pickle
import tracemalloc

import bm25s

from bm25_tokenize import tokenize_for_bm25

INDEX_DIR   = os.path.join(os.path.dirname(__file__), "../../indexes")
CORPUS_PATH = os.path.join(INDEX_DIR, "corpus.pkl")
BM25_DIR    = os.path.join(INDEX_DIR, "bm25")


def main() -> None:
    os.makedirs(BM25_DIR, exist_ok=True)

    # Load corpus
    print("Loading corpus.pkl ...")
    with open(CORPUS_PATH, "rb") as f:
        corpus: list[dict] = pickle.load(f)
    print(f"Corpus size: {len(corpus):,} documents")

    indexed_texts = [doc["indexed_text"] for doc in corpus]
    citations     = [doc["citation"] for doc in corpus]

    print("Tokenizing (lowercase, German stopwords, Snowball stem, citation tokens) ...")
    tracemalloc.start()
    tokenized = tokenize_for_bm25(indexed_texts, citations=citations, show_progress=True)

    # Build BM25 index
    print("Building BM25 index ...")
    retriever = bm25s.BM25()
    retriever.index(tokenized, show_progress=True)

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"Peak memory during indexing: {peak / 1e9:.2f} GB")

    # Save index
    retriever.save(BM25_DIR, corpus=indexed_texts)
    print(f"BM25 index saved → {BM25_DIR}/")


if __name__ == "__main__":
    main()
