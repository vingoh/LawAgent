# Reranker Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Insert `bge-reranker-v2-m3` as a cross-encoder reranking step after hybrid RRF, and extract corpus loading into a shared module to eliminate duplicate memory usage.

**Architecture:** New `corpus.py` becomes the single loader for `corpus.pkl`; `bm25.py` and `dense.py` delegate corpus loading to it. New `rerank.py` wraps `FlagReranker` with lazy-loading and batch inference. `run.py` calls `rerank()` between RRF fusion and the final `[:k]` slice.

**Tech Stack:** Python 3.11, FlagEmbedding (`FlagReranker`), bm25s, sentence-transformers, FAISS, pytest

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/retrieval/corpus.py` | Single loader for corpus.pkl; exposes court/law lists and citation→text map |
| Create | `src/retrieval/rerank.py` | FlagReranker lazy-load + batch `rerank()` |
| Create | `tests/retrieval/test_corpus.py` | Unit tests for corpus module |
| Create | `tests/retrieval/test_rerank.py` | Unit tests for rerank module (mocked model) |
| Modify | `src/retrieval/bm25.py` | Remove corpus pickle loading; use corpus module |
| Modify | `src/retrieval/dense.py` | Remove corpus pickle loading; use corpus module |
| Modify | `src/query/run.py` | Integrate reranker; new CLI args; updated startup sequence |

---

## Task 1: Create `src/retrieval/corpus.py`

**Files:**
- Create: `src/retrieval/corpus.py`
- Create: `tests/retrieval/test_corpus.py`

- [ ] **Step 1: Write failing tests**

Create `tests/retrieval/test_corpus.py`:

```python
import os
import sys
import pickle
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import retrieval.corpus as corpus_mod


def _make_corpus(tmp_path: str) -> str:
    """Write a minimal corpus.pkl for tests."""
    docs = [
        {"citation": "Art. 1 OR",  "source": "law",   "indexed_text": "text law 1"},
        {"citation": "Art. 2 OR",  "source": "law",   "indexed_text": "text law 2"},
        {"citation": "BGE 1 I 1",  "source": "court", "indexed_text": "text court 1"},
    ]
    path = os.path.join(tmp_path, "corpus.pkl")
    with open(path, "wb") as f:
        pickle.dump(docs, f)
    return path


def _reset_corpus():
    """Reset module-level state between tests."""
    corpus_mod._corpus_court = None
    corpus_mod._corpus_law = None
    corpus_mod._corpus_texts = {}


def test_load_corpus_splits_by_source(tmp_path, monkeypatch):
    _reset_corpus()
    corpus_path = _make_corpus(str(tmp_path))
    monkeypatch.setattr(corpus_mod, "CORPUS_PATH", corpus_path)

    corpus_mod.load_corpus()

    assert len(corpus_mod.get_corpus_court()) == 1
    assert len(corpus_mod.get_corpus_law()) == 2
    assert corpus_mod.get_corpus_court()[0]["citation"] == "BGE 1 I 1"


def test_load_corpus_is_idempotent(tmp_path, monkeypatch):
    _reset_corpus()
    corpus_path = _make_corpus(str(tmp_path))
    monkeypatch.setattr(corpus_mod, "CORPUS_PATH", corpus_path)

    corpus_mod.load_corpus()
    court_id = id(corpus_mod._corpus_court)
    corpus_mod.load_corpus()  # second call
    assert id(corpus_mod._corpus_court) == court_id  # same object, not reloaded


def test_get_corpus_texts(tmp_path, monkeypatch):
    _reset_corpus()
    corpus_path = _make_corpus(str(tmp_path))
    monkeypatch.setattr(corpus_mod, "CORPUS_PATH", corpus_path)

    corpus_mod.load_corpus()
    texts = corpus_mod.get_corpus_texts()

    assert texts["Art. 1 OR"] == "text law 1"
    assert texts["Art. 2 OR"] == "text law 2"
    assert texts["BGE 1 I 1"] == "text court 1"


def test_get_corpus_texts_is_cached(tmp_path, monkeypatch):
    _reset_corpus()
    corpus_path = _make_corpus(str(tmp_path))
    monkeypatch.setattr(corpus_mod, "CORPUS_PATH", corpus_path)

    corpus_mod.load_corpus()
    t1 = corpus_mod.get_corpus_texts()
    t2 = corpus_mod.get_corpus_texts()
    assert t1 is t2  # same dict object


def test_load_corpus_missing_file(monkeypatch):
    _reset_corpus()
    monkeypatch.setattr(corpus_mod, "CORPUS_PATH", "/nonexistent/corpus.pkl")
    try:
        corpus_mod.load_corpus()
        assert False, "Should have raised"
    except FileNotFoundError:
        pass
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n agent pytest tests/retrieval/test_corpus.py -v
```

Expected: `ModuleNotFoundError` or `AttributeError` (corpus.py doesn't exist yet).

- [ ] **Step 3: Implement `src/retrieval/corpus.py`**

```python
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
_corpus_texts: dict[str, str]   = {}


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
    if _corpus_texts:
        return _corpus_texts
    court = get_corpus_court()
    law   = get_corpus_law()
    _corpus_texts = {d["citation"]: d["indexed_text"] for d in court + law}
    return _corpus_texts
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n agent pytest tests/retrieval/test_corpus.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/corpus.py tests/retrieval/test_corpus.py
git commit -m "feat: add shared corpus loader module"
```

---

## Task 2: Refactor `src/retrieval/bm25.py` to use corpus module

**Files:**
- Modify: `src/retrieval/bm25.py`

`bm25.py` currently loads `corpus.pkl` itself in `_load_index()` (lines 57–65). Replace that block with a call to the corpus module.

- [ ] **Step 1: Edit `src/retrieval/bm25.py`**

Add the import at the top (after existing imports):

```python
from retrieval import corpus as _corpus_mod
```

Replace the `_load_index()` function body. Current version:

```python
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
```

Replace with:

```python
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
```

Also remove the now-unused import and constant:

```python
# Remove these lines:
import pickle                   # no longer needed in bm25.py
CORPUS_PATH = ...               # no longer needed in bm25.py
```

- [ ] **Step 2: Verify no tests broken**

```bash
conda run -n agent pytest tests/ -v
```

Expected: all previously passing tests still PASS.

- [ ] **Step 3: Commit**

```bash
git add src/retrieval/bm25.py
git commit -m "refactor: bm25 delegates corpus loading to corpus module"
```

---

## Task 3: Refactor `src/retrieval/dense.py` to use corpus module

**Files:**
- Modify: `src/retrieval/dense.py`

`dense.py` loads `corpus.pkl` in `_load_index()` (lines 77–81). Replace with corpus module.

- [ ] **Step 1: Edit `src/retrieval/dense.py`**

Add import after existing imports:

```python
from retrieval import corpus as _corpus_mod
```

In `_load_index()`, replace the corpus-loading block:

```python
    # Current (remove this):
    print("Loading corpus.pkl for dense …", file=sys.stderr)
    with open(CORPUS_PATH, "rb") as f:
        corpus = pickle.load(f)
    _corpus_court = [d for d in corpus if d["source"] == "court"]
    _corpus_law   = [d for d in corpus if d["source"] == "law"]
    _loaded = True
```

With:

```python
    # Replacement:
    _corpus_court = _corpus_mod.get_corpus_court()
    _corpus_law   = _corpus_mod.get_corpus_law()
    _loaded = True
```

Also remove the now-unused constant and import:

```python
# Remove these lines:
import pickle               # no longer needed in dense.py
CORPUS_PATH = INDEX_DIR / "corpus.pkl"   # no longer needed in dense.py
```

- [ ] **Step 2: Verify no tests broken**

```bash
conda run -n agent pytest tests/ -v
```

Expected: all previously passing tests still PASS.

- [ ] **Step 3: Commit**

```bash
git add src/retrieval/dense.py
git commit -m "refactor: dense delegates corpus loading to corpus module"
```

---

## Task 4: Create `src/retrieval/rerank.py`

**Files:**
- Create: `src/retrieval/rerank.py`
- Create: `tests/retrieval/test_rerank.py`

- [ ] **Step 1: Write failing tests**

Create `tests/retrieval/test_rerank.py`:

```python
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import retrieval.rerank as rerank_mod


CORPUS_TEXTS = {
    "Art. 1 OR":  "Erster Artikel des OR",
    "Art. 2 OR":  "Zweiter Artikel des OR",
    "BGE 1 I 1":  "Bundesgericht Entscheid",
    "Art. 3 OR":  "Dritter Artikel",
}


def _reset_reranker():
    rerank_mod._reranker = None


def _mock_reranker(scores: list[float]):
    """Return a mock FlagReranker that returns the given scores."""
    mock = MagicMock()
    mock.compute_score.return_value = scores
    return mock


def test_rerank_reorders_by_score():
    _reset_reranker()
    # Scores: Art.2 highest (0.9), Art.1 middle (0.5), BGE lowest (0.1)
    mock = _mock_reranker([0.5, 0.9, 0.1])
    rerank_mod._reranker = mock

    candidates = ["Art. 1 OR", "Art. 2 OR", "BGE 1 I 1"]
    result = rerank_mod.rerank(
        "test query", candidates, CORPUS_TEXTS, top_k=3, batch_size=32
    )

    assert result[0] == "Art. 2 OR"
    assert result[1] == "Art. 1 OR"
    assert result[2] == "BGE 1 I 1"


def test_rerank_appends_tail_beyond_top_k():
    _reset_reranker()
    # Only top_k=2 candidates are sent to reranker; Art.3 is tail
    mock = _mock_reranker([0.8, 0.3])  # scores for Art.1, Art.2
    rerank_mod._reranker = mock

    candidates = ["Art. 1 OR", "Art. 2 OR", "Art. 3 OR"]
    result = rerank_mod.rerank(
        "test query", candidates, CORPUS_TEXTS, top_k=2, batch_size=32
    )

    assert result[:2] == ["Art. 1 OR", "Art. 2 OR"]  # reranked top-2
    assert result[2] == "Art. 3 OR"   # tail preserved in RRF order


def test_rerank_missing_text_gets_neg_inf():
    _reset_reranker()
    # "Art. 2 OR" has no text in corpus_texts
    mock = _mock_reranker([0.7])  # score only for Art.1 (Art.2 has no text)
    rerank_mod._reranker = mock

    sparse_corpus = {"Art. 1 OR": "some text"}
    candidates = ["Art. 1 OR", "Art. 2 OR"]
    result = rerank_mod.rerank(
        "test query", candidates, sparse_corpus, top_k=2, batch_size=32
    )

    # Art.1 scored 0.7, Art.2 scored -inf → Art.1 first
    assert result[0] == "Art. 1 OR"
    assert result[1] == "Art. 2 OR"


def test_rerank_empty_candidates():
    _reset_reranker()
    rerank_mod._reranker = MagicMock()

    result = rerank_mod.rerank("query", [], CORPUS_TEXTS, top_k=100)
    assert result == []
    rerank_mod._reranker.compute_score.assert_not_called()


def test_reranker_exists_false_when_no_model(monkeypatch, tmp_path):
    monkeypatch.setattr(rerank_mod, "RERANKER_PATH", str(tmp_path / "nonexistent"))
    assert rerank_mod.reranker_exists() is False


def test_reranker_exists_true_when_model_present(monkeypatch, tmp_path):
    model_dir = tmp_path / "bge-reranker-v2-m3"
    model_dir.mkdir()
    monkeypatch.setattr(rerank_mod, "RERANKER_PATH", str(model_dir))
    assert rerank_mod.reranker_exists() is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
conda run -n agent pytest tests/retrieval/test_rerank.py -v
```

Expected: `ModuleNotFoundError` (rerank.py doesn't exist yet).

- [ ] **Step 3: Implement `src/retrieval/rerank.py`**

```python
"""Cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Usage (from run.py):
    from retrieval.rerank import reranker_exists, _load_reranker, rerank

    if reranker_exists():
        _load_reranker()

    reranked = rerank(query, rrf_result, corpus_texts, top_k=100)
"""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RERANKER_PATH = str(ROOT_DIR / "models" / "bge-reranker-v2-m3")

_reranker = None  # FlagReranker instance, lazy-loaded


def reranker_exists() -> bool:
    """Return True if the reranker model directory exists on disk."""
    return os.path.isdir(RERANKER_PATH)


def _load_reranker() -> None:
    """Lazy-load FlagReranker. Idempotent — no-op if already loaded."""
    global _reranker
    if _reranker is not None:
        return
    from FlagEmbedding import FlagReranker  # imported lazily to avoid startup cost

    print(f"Loading reranker from {RERANKER_PATH} ...", file=sys.stderr)
    _reranker = FlagReranker(RERANKER_PATH, use_fp16=True)
    print("Reranker ready.", file=sys.stderr)


def rerank(
    query: str,
    candidates: list[str],
    corpus_texts: dict[str, str],
    *,
    top_k: int = 100,
    batch_size: int = 32,
) -> list[str]:
    """Rerank the top_k candidates using the cross-encoder, preserving the tail.

    Args:
        query:        Query string sent to the reranker (typically raw_query +
                      " " + search_text).
        candidates:   RRF-ordered citation list (full length).
        corpus_texts: citation → indexed_text mapping from corpus module.
        top_k:        Number of candidates to score. candidates[top_k:] are
                      appended after the reranked head in their original order.
        batch_size:   Batch size for reranker inference.

    Returns:
        Full citation list: reranked head (candidates[:top_k]) followed by
        the unchanged tail (candidates[top_k:]).
        Citations missing from corpus_texts receive score -inf and appear
        last within the reranked head.
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
        scores: list[float] = _reranker.compute_score(
            pairs, batch_size=batch_size, normalize=True
        )
    else:
        scores = []

    # Sort by score descending.
    scored = sorted(zip(scored_citations, scores), key=lambda x: x[1], reverse=True)
    reranked_head = [cit for cit, _ in scored]

    # Append citations with missing text (in original order) after scored ones.
    missing_ordered = [cit for cit in head if cit in no_text]
    reranked_head.extend(missing_ordered)

    return reranked_head + tail
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
conda run -n agent pytest tests/retrieval/test_rerank.py -v
```

Expected: 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/rerank.py tests/retrieval/test_rerank.py
git commit -m "feat: add reranker module with FlagReranker lazy-loading"
```

---

## Task 5: Integrate reranker into `src/query/run.py`

**Files:**
- Modify: `src/query/run.py`

This task makes 6 targeted changes to `run.py`. Apply them in order.

- [ ] **Step 1: Update imports at the top of `run.py`**

Replace the current import block (lines 17–25):

```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from query.rewrite import format_search_text, rewrite_query
from retrieval.bm25 import _load_index as _load_bm25_index, retrieve_bm25_parts
from retrieval.dense import (
    _load_index as _load_dense_index,
    dense_court_exists,
    dense_law_exists,
    retrieve_dense_parts,
)
from retrieval.rrf import weighted_rrf
```

With:

```python
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
from retrieval.rerank import _load_reranker, rerank, reranker_exists
from retrieval.rrf import weighted_rrf
```

- [ ] **Step 2: Add `_USE_RERANK` module-level flag**

After the existing module-level flags (after line 33: `_USE_DENSE_LAW: bool = dense_law_exists()`), add:

```python
_USE_RERANK: bool = reranker_exists()
```

- [ ] **Step 3: Update `_print_run_config()` to show reranker status**

Add two parameters to the function signature and one print line. Replace:

```python
def _print_run_config(
    input_path: str,
    output_path: str,
    *,
    use_rewrite: bool,
    rewrite_log_dir: str | None,
) -> None:
    print("Run configuration:", file=sys.stderr)
    print(f"  ROOT_DIR                = {ROOT_DIR}", file=sys.stderr)
    print(f"  DATASET_DIR             = {DATASET_DIR}", file=sys.stderr)
    print(f"  DEFAULT_INPUT           = {DEFAULT_INPUT}", file=sys.stderr)
    print(f"  DEFAULT_OUTPUT          = {DEFAULT_OUTPUT}", file=sys.stderr)
    print(f"  DEFAULT_REWRITE_LOG_DIR = {DEFAULT_REWRITE_LOG_DIR}", file=sys.stderr)
    print(f"  dense_court             = {_USE_DENSE_COURT}", file=sys.stderr)
    print(f"  dense_law               = {_USE_DENSE_LAW}", file=sys.stderr)
    print(f"  input (actual)          = {input_path}", file=sys.stderr)
    print(f"  output (actual)         = {output_path}", file=sys.stderr)
    print(f"  use_rewrite             = {use_rewrite}", file=sys.stderr)
    print(f"  rewrite_log_dir         = {rewrite_log_dir}", file=sys.stderr)
```

With:

```python
def _print_run_config(
    input_path: str,
    output_path: str,
    *,
    use_rewrite: bool,
    rewrite_log_dir: str | None,
    use_rerank: bool,
    rerank_top_k: int,
    rerank_batch_size: int,
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
    print(f"  input (actual)          = {input_path}", file=sys.stderr)
    print(f"  output (actual)         = {output_path}", file=sys.stderr)
    print(f"  use_rewrite             = {use_rewrite}", file=sys.stderr)
    print(f"  rewrite_log_dir         = {rewrite_log_dir}", file=sys.stderr)
```

- [ ] **Step 4: Update `predict_citations()` to call reranker after RRF**

Add three new keyword parameters and the rerank logic. Replace:

```python
def predict_citations(
    query: str,
    *,
    use_rewrite: bool = True,
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
```

With:

```python
def predict_citations(
    query: str,
    *,
    use_rewrite: bool = True,
    use_rerank: bool = True,
    rerank_top_k: int = 100,
    rerank_batch_size: int = 32,
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
```

Then replace the final `return` line:

```python
    return weighted_rrf(rankings, rrf_k=rrf_k)[:k]
```

With:

```python
    rrf_result = weighted_rrf(rankings, rrf_k=rrf_k)

    if use_rerank and _USE_RERANK:
        rerank_query = query
        if search_text is not None:
            rerank_query = query + " " + search_text
        rrf_result = rerank(
            rerank_query,
            rrf_result,
            corpus.get_corpus_texts(),
            top_k=rerank_top_k,
            batch_size=rerank_batch_size,
        )

    return rrf_result[:k]
```

- [ ] **Step 5: Update `_process_query()` to pass reranker params**

Replace the `_process_query()` signature and body:

```python
def _process_query(
    row: dict[str, str],
    *,
    use_rewrite: bool,
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
```

With:

```python
def _process_query(
    row: dict[str, str],
    *,
    use_rewrite: bool,
    use_rerank: bool,
    rerank_top_k: int,
    rerank_batch_size: int,
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
```

- [ ] **Step 6: Update `run()` function**

Replace the `run()` signature:

```python
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
    rewrite_log_dir: str | None = None,
    workers: int = 4,
) -> None:
```

With:

```python
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
    rewrite_log_dir: str | None = None,
    workers: int = 4,
) -> None:
```

Update the `_print_run_config()` call inside `run()`:

```python
    _print_run_config(
        input_path,
        output_path,
        use_rewrite=use_rewrite,
        rewrite_log_dir=rewrite_log_dir,
        use_rerank=use_rerank,
        rerank_top_k=rerank_top_k,
        rerank_batch_size=rerank_batch_size,
    )
```

Update the index loading block inside `run()`. Replace:

```python
    _load_bm25_index()
    if _USE_DENSE_COURT or _USE_DENSE_LAW:
        _load_dense_index(use_court=_USE_DENSE_COURT, use_law=_USE_DENSE_LAW)
```

With:

```python
    corpus.load_corpus()
    _load_bm25_index()
    if _USE_DENSE_COURT or _USE_DENSE_LAW:
        _load_dense_index(use_court=_USE_DENSE_COURT, use_law=_USE_DENSE_LAW)
    if use_rerank and _USE_RERANK:
        _load_reranker()
```

Update `process_kwargs` dict inside `run()`. Add three new keys:

```python
    process_kwargs = {
        "use_rewrite": use_rewrite,
        "use_rerank": use_rerank,
        "rerank_top_k": rerank_top_k,
        "rerank_batch_size": rerank_batch_size,
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
```

- [ ] **Step 7: Update `main()` to add CLI arguments**

Inside `main()`, after the existing `--no-rewrite` argument block, add:

```python
    parser.add_argument("--no-rerank", action="store_true",
                        help="Skip reranker (default: reranker ON if model exists)")
    parser.add_argument("--rerank-top-k", type=int, default=100,
                        help="Number of RRF candidates sent to reranker (default: 100)")
    parser.add_argument("--rerank-batch-size", type=int, default=32,
                        help="Reranker inference batch size (default: 32)")
```

Update the `run()` call in `main()`:

```python
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
        rewrite_log_dir=None if args.no_rewrite_log else args.rewrite_log,
        workers=args.workers,
    )
```

- [ ] **Step 8: Run all tests**

```bash
conda run -n agent pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 9: Commit**

```bash
git add src/query/run.py
git commit -m "feat: integrate reranker into predict_citations pipeline"
```

---

## Task 6: Smoke test end-to-end

This task verifies the full pipeline runs without errors. It does not require a specific F1 score — we just confirm the pipeline starts, loads models, runs queries, and writes predictions.

- [ ] **Step 1: Run pipeline with `--no-rerank` (baseline, fast)**

```bash
conda run -n agent python src/query/run.py \
    --input dataset/val.csv \
    --output results/predictions_no_rerank.csv \
    --no-rerank \
    --workers 1
```

Expected: runs to completion, prints `Wrote N predictions → results/predictions_no_rerank.csv`.

- [ ] **Step 2: Run pipeline with reranker enabled**

```bash
conda run -n agent python src/query/run.py \
    --input dataset/val.csv \
    --output results/predictions_rerank.csv \
    --workers 1
```

Expected:
- Stderr shows `Loading reranker from .../models/bge-reranker-v2-m3 ...`
- Stderr shows `Reranker ready.`
- Stderr shows `reranker = True (top_k=100, batch=32)`
- Runs to completion and writes predictions CSV.

- [ ] **Step 3: Evaluate both outputs**

```bash
conda run -n agent python src/eval/macro_f1.py \
    --predictions results/predictions_no_rerank.csv

conda run -n agent python src/eval/macro_f1.py \
    --predictions results/predictions_rerank.csv
```

Expected: both commands print a Macro F1 score without error. The reranker run should show equal or higher F1.

- [ ] **Step 4: Final commit**

```bash
git add results/predictions_no_rerank.csv results/predictions_rerank.csv || true
git commit -m "feat: reranker integration complete — smoke test passed"
```

(The results CSVs are gitignored; the commit may be empty or just update the plan — that is fine.)

---

## Self-Review Checklist

**Spec coverage:**

| Spec requirement | Covered by |
|-----------------|-----------|
| New `corpus.py` shared loader | Task 1 |
| `bm25.py` delegates corpus loading | Task 2 |
| `dense.py` delegates corpus loading | Task 3 |
| New `rerank.py` with FlagReranker | Task 4 |
| `rerank()` takes top-100 RRF candidates | Task 4 (top_k=100 default) |
| Reranker query = raw_query + " " + search_text | Task 5 Step 4 |
| Final cutoff still controlled by `--k` | Task 5 Step 4 (return rrf_result[:k]) |
| Default ON, `--no-rerank` to disable | Task 5 Step 7 |
| Graceful degradation when model missing | Task 4 (`reranker_exists()`), Task 5 (`_USE_RERANK` flag) |
| Missing-text citations get score -inf | Task 4 implementation |
| Tail beyond top_k preserved in RRF order | Task 4 implementation |
| `_print_run_config()` shows reranker status | Task 5 Step 3 |
| `--rerank-top-k`, `--rerank-batch-size` CLI args | Task 5 Step 7 |

**Placeholder scan:** No TBD, TODO, or vague steps — all steps contain complete code.

**Type consistency:**
- `rerank()` signature: `(query: str, candidates: list[str], corpus_texts: dict[str, str], *, top_k: int, batch_size: int) -> list[str]` — consistent across Task 4 tests, Task 4 implementation, and Task 5 call site.
- `corpus.get_corpus_texts()` returns `dict[str, str]` — consistent across Task 1 and Task 5.
- `reranker_exists()` returns `bool` — consistent with Task 4 tests and Task 5 `_USE_RERANK` assignment.
