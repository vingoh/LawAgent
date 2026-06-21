"""Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Usage (from run.py):
    from retrieval.rerank import reranker_exists, rerank_with_scores, rerank

    if reranker_exists():
        _load_reranker()  # optional pre-warm; functions call it automatically

    scored = rerank_with_scores(query, rrf_result, corpus_texts, top_k=100)
    reranked = rerank(query, rrf_result, corpus_texts, top_k=100)  # backward compat
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RERANKER_PATH = str(ROOT_DIR / "models" / "bge-reranker-v2-m3")

_reranker_model = None   # AutoModelForSequenceClassification
_reranker_tok = None     # AutoTokenizer
_reranker_device = None  # torch.device


def reranker_exists() -> bool:
    """Return True if the reranker model directory exists on disk."""
    return os.path.isdir(RERANKER_PATH)


def _load_reranker() -> None:
    """Lazy-load the cross-encoder model via transformers. Idempotent."""
    global _reranker_model, _reranker_tok, _reranker_device
    if _reranker_model is not None:
        return

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"Loading reranker from {RERANKER_PATH} ...", file=sys.stderr)
    _reranker_tok = AutoTokenizer.from_pretrained(RERANKER_PATH)
    _reranker_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _reranker_model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_PATH, dtype=torch.float16
    ).to(_reranker_device)
    _reranker_model.eval()
    print("Reranker ready.", file=sys.stderr)


def _compute_scores(
    pairs: list[list[str]], batch_size: int = 32
) -> list[float]:
    """Score (query, passage) pairs with the cross-encoder; return sigmoid scores."""
    import torch

    all_scores: list[float] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        enc = _reranker_tok(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(_reranker_device)
        with torch.no_grad():
            logits = _reranker_model(**enc).logits.squeeze(-1)
        scores = torch.sigmoid(logits).float().tolist()
        if isinstance(scores, float):
            scores = [scores]
        all_scores.extend(scores)
    return all_scores


def rerank_with_scores(
    query: str,
    candidates: list[str],
    corpus_texts: dict[str, str],
    *,
    top_k: int = 100,
    batch_size: int = 32,
) -> list[tuple[str, float]]:
    """Score and rerank top_k candidates using cross-encoder.

    Args:
        query:        Query string.
        candidates:   Citation list (only candidates[:top_k] are scored).
        corpus_texts: citation -> indexed_text mapping.
        top_k:        Number of candidates to score.
        batch_size:   Batch size for reranker inference.

    Returns:
        List of (citation, score) tuples for candidates[:top_k], sorted by
        score descending. Citations missing from corpus_texts receive score 0.0
        and appear at the end.
    """
    if not candidates:
        return []

    _load_reranker()

    head = candidates[:top_k]

    pairs: list[list[str]] = []
    no_text: set[str] = set()
    for cit in head:
        text = corpus_texts.get(cit)
        if text is None:
            no_text.add(cit)
        else:
            pairs.append([query, text])

    scored_citations = [cit for cit in head if cit not in no_text]
    if scored_citations:
        raw_scores: list[float] = _compute_scores(pairs, batch_size)
    else:
        raw_scores = []

    scored = sorted(
        zip(scored_citations, raw_scores), key=lambda x: x[1], reverse=True
    )
    missing = [(cit, 0.0) for cit in head if cit in no_text]
    return list(scored) + missing


def rerank(
    query: str,
    candidates: list[str],
    corpus_texts: dict[str, str],
    *,
    top_k: int = 100,
    batch_size: int = 32,
) -> list[str]:
    """Rerank top_k candidates using cross-encoder; preserve the tail unchanged.

    Backward-compatible wrapper around rerank_with_scores(). Returns list[str].
    candidates[top_k:] are appended after the reranked head in original order.
    """
    if not candidates:
        return candidates

    scored = rerank_with_scores(
        query, candidates, corpus_texts, top_k=top_k, batch_size=batch_size
    )
    reranked_head = [cit for cit, _ in scored]

    tail = candidates[top_k:]
    return reranked_head + tail
