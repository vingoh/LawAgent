"""Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Usage (from run.py):
    from retrieval.rerank import reranker_exists, rerank

    if reranker_exists():
        _load_reranker()  # optional pre-warm; rerank() calls it automatically

    reranked = rerank(query, rrf_result, corpus_texts, top_k=100)
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


def rerank(
    query: str,
    candidates: list[str],
    corpus_texts: dict[str, str],
    *,
    top_k: int = 100,
    batch_size: int = 32,
) -> list[str]:
    """Rerank top_k candidates using cross-encoder; preserve the tail unchanged.

    Args:
        query:        Query string sent to the reranker (typically raw_query +
                      " " + search_text).
        candidates:   RRF-ordered citation list (full length).
        corpus_texts: citation -> indexed_text mapping from corpus module.
        top_k:        Number of candidates to score. candidates[top_k:] are
                      appended after the reranked head in their original order.
        batch_size:   Batch size for reranker inference.

    Returns:
        Full citation list: reranked head (candidates[:top_k]) followed by
        the unchanged tail (candidates[top_k:]).
        Citations missing from corpus_texts appear last within the reranked head.
    """
    if not candidates:
        return candidates

    _load_reranker()

    head = candidates[:top_k]
    tail = candidates[top_k:]

    # Build (query, passage) pairs; track which citations have no text.
    pairs: list[list[str]] = []
    no_text: set[str] = set()
    for cit in head:
        text = corpus_texts.get(cit)
        if text is None:
            no_text.add(cit)
        else:
            pairs.append([query, text])

    # Score only citations that have text.
    scored_citations = [cit for cit in head if cit not in no_text]
    if scored_citations:
        scores: list[float] = _compute_scores(pairs, batch_size=batch_size)
    else:
        scores = []

    # Sort by score descending.
    scored = sorted(zip(scored_citations, scores), key=lambda x: x[1], reverse=True)
    reranked_head = [cit for cit, _ in scored]

    # Append citations with missing text (in original order) after scored ones.
    missing_ordered = [cit for cit in head if cit in no_text]
    reranked_head.extend(missing_ordered)

    return reranked_head + tail
