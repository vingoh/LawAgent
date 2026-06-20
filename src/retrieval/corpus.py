"""Shared corpus loader.

Loads corpus.pkl exactly once for the lifetime of the process.
All retrieval modules (bm25, dense, rerank) use this module instead of
loading corpus.pkl independently.
"""

import os
import pickle
import sys
from pathlib import Path

ROOT_DIR    = Path(__file__).resolve().parents[2]
CORPUS_PATH = str(ROOT_DIR / "indexes" / "corpus.pkl")

_corpus_court: list[dict] | None = None
_corpus_law:   list[dict] | None = None
_corpus_texts: dict[str, str] | None = None


def load_corpus() -> None:
    """Load corpus.pkl and split by source field. Idempotent."""
    global _corpus_court, _corpus_law
    if _corpus_court is not None:
        return
    if not os.path.exists(CORPUS_PATH):
        raise FileNotFoundError(
            f"corpus.pkl not found: {CORPUS_PATH}. "
            "Run: conda run -n agent python src/indexing/build_corpus.py"
        )
    print("Loading corpus.pkl ...", file=sys.stderr)
    with open(CORPUS_PATH, "rb") as f:
        corpus: list[dict] = pickle.load(f)
    _corpus_court = [d for d in corpus if d["source"] == "court"]
    _corpus_law   = [d for d in corpus if d["source"] == "law"]
    print(
        f"Corpus ready. Court: {len(_corpus_court):,}, law: {len(_corpus_law):,}",
        file=sys.stderr,
    )


def get_corpus_court() -> list[dict]:
    """Return list of court documents. Calls load_corpus() if needed."""
    if _corpus_court is None:
        load_corpus()
    return _corpus_court  # type: ignore[return-value]


def get_corpus_law() -> list[dict]:
    """Return list of law documents. Calls load_corpus() if needed."""
    if _corpus_law is None:
        load_corpus()
    return _corpus_law  # type: ignore[return-value]


def get_corpus_texts() -> dict[str, str]:
    """Return citation→indexed_text mapping. Built lazily on first call."""
    global _corpus_texts
    if _corpus_texts is not None:
        return _corpus_texts
    court = get_corpus_court()
    law   = get_corpus_law()
    _corpus_texts = {d["citation"]: d["indexed_text"] for d in court + law}
    return _corpus_texts
