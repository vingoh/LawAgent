# Selector Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a 5-stage post-reranker selector pipeline (`cheap_expand → llm_verify → fuse_scores → adaptive_count → assemble`) that replaces the fixed top-k truncation with adaptive, semantically verified citation selection.

**Architecture:** New `src/retrieval/selector.py` holds the `Candidate` dataclass and all pipeline stages, exposed via `run_selector()`. `rrf.py` is updated to also return a score dict. `rerank.py` gets a new `rerank_with_scores()` function. `run.py` wires everything together.

**Tech Stack:** Python 3.11, `kneed>=0.8` (Kneedle elbow detection), OpenAI-compatible LLM client (`llm/client.py`), pytest

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/retrieval/rrf.py` | Return `(ordered_list, scores_dict)` tuple |
| Modify | `src/retrieval/rerank.py` | Add `rerank_with_scores()` returning `list[tuple[str, float]]` |
| Create | `src/retrieval/selector.py` | `Candidate` dataclass + `build_candidates` + all 5 pipeline stages + `run_selector` |
| Modify | `src/query/run.py` | Wire new interfaces; add `--no-select`, `--no-llm-verify`, `--verifier-top-k` CLI args |
| Modify | `tests/retrieval/test_rrf.py` | Update tests to unpack tuple return value |
| Modify | `tests/retrieval/test_rerank.py` | Add tests for `rerank_with_scores` |
| Create | `tests/retrieval/test_selector.py` | Unit tests for all selector stages |

---

## Task 1: Update `rrf.py` to return score dict

**Files:**
- Modify: `src/retrieval/rrf.py`
- Modify: `tests/retrieval/test_rrf.py`

- [ ] **Step 1: Update tests to expect tuple**

Replace the entire content of `tests/retrieval/test_rrf.py`:

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from retrieval.rrf import weighted_rrf


def test_single_ranking():
    result, scores = weighted_rrf([( ["a", "b", "c"], 1.0)], rrf_k=60)
    assert result == ["a", "b", "c"]
    assert set(scores.keys()) == {"a", "b", "c"}
    assert scores["a"] > scores["b"] > scores["c"]


def test_multi_ranking_accumulates():
    result, scores = weighted_rrf(
        [(["a", "b"], 1.0), (["b", "c"], 1.0)], rrf_k=60
    )
    assert result[0] == "b"
    assert set(result) == {"a", "b", "c"}


def test_weighted_ranking():
    result, scores = weighted_rrf(
        [(["a"], 1.0), (["b"], 3.0)], rrf_k=60
    )
    assert result[0] == "b"


def test_empty_rankings():
    result, scores = weighted_rrf([], rrf_k=60)
    assert result == []
    assert scores == {}


def test_empty_single_list():
    result, scores = weighted_rrf([([], 1.0), (["a"], 1.0)], rrf_k=60)
    assert result == ["a"]
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
conda run -n agent pytest tests/retrieval/test_rrf.py -v 2>&1 | tail -20
```

Expected: 5 failures (tuple unpacking errors on old return value).

- [ ] **Step 3: Update `src/retrieval/rrf.py`**

```python
"""Reciprocal Rank Fusion utilities."""


def weighted_rrf(
    rankings: list[tuple[list[str], float]],
    rrf_k: int = 60,
) -> tuple[list[str], dict[str, float]]:
    """Fuse multiple weighted rankings into a single ordered citation list.

    score(c) += weight / (rrf_k + rank + 1)  for each appearance in a ranking.

    Returns:
        (ordered_list, scores_dict) where scores_dict maps citation -> fused score.
    """
    scores: dict[str, float] = {}
    for ranked_list, weight in rankings:
        if not ranked_list or weight == 0:
            continue
        for rank, citation in enumerate(ranked_list):
            scores[citation] = scores.get(citation, 0.0) + weight / (rrf_k + rank + 1)

    ordered = sorted(scores, key=lambda c: scores[c], reverse=True)
    return ordered, scores
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
conda run -n agent pytest tests/retrieval/test_rrf.py -v 2>&1 | tail -10
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/rrf.py tests/retrieval/test_rrf.py
git commit -m "rrf: return (ordered_list, scores_dict) tuple"
```

---

## Task 2: Add `rerank_with_scores()` to `rerank.py`

**Files:**
- Modify: `src/retrieval/rerank.py`
- Modify: `tests/retrieval/test_rerank.py`

- [ ] **Step 1: Write failing test**

Append to `tests/retrieval/test_rerank.py`:

```python
def test_rerank_with_scores_returns_scored_pairs():
    _reset_reranker()
    with patch.object(rerank_mod, "_compute_scores", return_value=[0.5, 0.9, 0.1]):
        candidates = ["Art. 1 OR", "Art. 2 OR", "BGE 1 I 1"]
        result = rerank_mod.rerank_with_scores(
            "test query", candidates, CORPUS_TEXTS, top_k=3, batch_size=32
        )
    # Returns list of (citation, score) tuples sorted by score desc
    assert isinstance(result, list)
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in result)
    citations = [c for c, _ in result]
    assert citations[0] == "Art. 2 OR"   # highest score 0.9
    assert citations[1] == "Art. 1 OR"   # score 0.5
    assert citations[2] == "BGE 1 I 1"   # lowest 0.1


def test_rerank_with_scores_missing_text_gets_zero():
    _reset_reranker()
    sparse = {"Art. 1 OR": "text"}
    with patch.object(rerank_mod, "_compute_scores", return_value=[0.7]):
        result = rerank_mod.rerank_with_scores(
            "q", ["Art. 1 OR", "Art. 2 OR"], sparse, top_k=2
        )
    scored_map = dict(result)
    assert scored_map["Art. 1 OR"] == pytest.approx(0.7, abs=1e-4)
    assert scored_map["Art. 2 OR"] == 0.0


def test_rerank_still_returns_list_of_strings():
    """Original rerank() backward-compat: returns list[str]."""
    _reset_reranker()
    with patch.object(rerank_mod, "_compute_scores", return_value=[0.5, 0.9, 0.1]):
        candidates = ["Art. 1 OR", "Art. 2 OR", "BGE 1 I 1"]
        result = rerank_mod.rerank("test query", candidates, CORPUS_TEXTS, top_k=3)
    assert isinstance(result, list)
    assert isinstance(result[0], str)
    assert result[0] == "Art. 2 OR"
```

Add `import pytest` at the top of test_rerank.py.

- [ ] **Step 2: Run — expect FAIL**

```bash
conda run -n agent pytest tests/retrieval/test_rerank.py::test_rerank_with_scores_returns_scored_pairs -v 2>&1 | tail -10
```

Expected: FAIL with `AttributeError: module ... has no attribute 'rerank_with_scores'`.

- [ ] **Step 3: Implement `rerank_with_scores()` and refactor `rerank()`**

Replace `src/retrieval/rerank.py` with:

```python
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
    print("Sorted scores: ", [s for _, s in scored[:100]])

    tail = candidates[top_k:]
    return reranked_head + tail
```

- [ ] **Step 4: Run all rerank tests — expect PASS**

```bash
conda run -n agent pytest tests/retrieval/test_rerank.py -v 2>&1 | tail -15
```

Expected: all tests pass (including the 3 new ones).

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/rerank.py tests/retrieval/test_rerank.py
git commit -m "rerank: add rerank_with_scores() returning scored pairs; refactor rerank() to use it"
```

---

## Task 3: Create `selector.py` — Candidate, build_candidates, cheap_expand

**Files:**
- Create: `src/retrieval/selector.py`

- [ ] **Step 1: Create `src/retrieval/selector.py` with Candidate dataclass and helpers**

```python
"""Post-reranker selector pipeline.

Pipeline order (all functions are in this module):
    build_candidates() → cheap_expand() → llm_verify() → fuse_scores()
    → adaptive_count() → assemble()

Entry point: run_selector()
"""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm.client import chat_json

_BGE_PARENT_RE = re.compile(
    r"^(BGE\s+\d+\s+(?:I{1,4}|IV|V)\s+\d+)(?:\s+E\..*)?$", re.IGNORECASE
)
_LAW_CODE_RE = re.compile(
    r"\b(StPO|StGB|OR|ZGB|ZPO|BGG|BV|ATSG|IVG|AHVG|UVG|AsylG|AIG|SchKG|DSG)\b"
)


@dataclass
class Candidate:
    # ── Core fields ───────────────────────────────────────────────────────
    citation: str
    rerank_score: float          # bge-reranker sigmoid [0, 1]; 0.0 for rescue candidates
    rrf_score: float             # weighted_rrf fused score

    # ── Metadata (filled by build_candidates / cheap_expand) ──────────────
    source_set: frozenset        # retrieval paths that found this citation
    direct_regex_hit: bool       # True if citation was in the extracted channel
    expected_code_match: bool    # True if citation's law code is in expected_codes
    expansion_type: str | None   # None / "same_parent" / "same_code" / "<src>_rescue"

    # ── LLM verifier output (None when verifier is skipped) ───────────────
    relevance: int | None = None       # 0-3
    llm_reason: str | None = None

    # ── Score fusion output ───────────────────────────────────────────────
    final_score: float = 0.0


def _extract_bge_parent(citation: str) -> str | None:
    """Return the BGE parent citation (without Erwägung), or None."""
    m = _BGE_PARENT_RE.match(citation)
    return m.group(1) if m else None


def _extract_law_code(citation: str) -> str | None:
    """Return the Swiss law code abbreviation from a citation, or None."""
    m = _LAW_CODE_RE.search(citation)
    return m.group(1) if m else None


def _is_seed(c: Candidate, threshold: float = 0.5) -> bool:
    return c.rerank_score >= threshold or c.direct_regex_hit
```

- [ ] **Step 2: Add `build_candidates()` to `selector.py`**

Append to `src/retrieval/selector.py`:

```python
def build_candidates(
    scored: list[tuple[str, float]],
    rrf_scores: dict[str, float],
    source_rankings: dict[str, list[str]],
    query_info,                              # RewriteResult or None
) -> list[Candidate]:
    """Build Candidate objects from reranker-scored results.

    Args:
        scored:          Output of rerank_with_scores(): [(citation, score), ...]
        rrf_scores:      Output of weighted_rrf(): citation -> rrf_score dict
        source_rankings: Per-retrieval-path citation lists
                         {"extracted": [...], "bm25_court": [...], ...}
        query_info:      RewriteResult (for expected_codes); may be None
    """
    source_sets = {src: set(ranking) for src, ranking in source_rankings.items()}
    extracted_set: set[str] = source_sets.get("extracted", set())
    expected_codes: set[str] = (
        set(query_info.expected_codes) if query_info is not None else set()
    )

    candidates: list[Candidate] = []
    for citation, rerank_score in scored:
        sources = frozenset(src for src, s in source_sets.items() if citation in s)
        direct = citation in extracted_set
        code = _extract_law_code(citation)
        code_match = code is not None and code in expected_codes
        candidates.append(
            Candidate(
                citation=citation,
                rerank_score=rerank_score,
                rrf_score=rrf_scores.get(citation, 0.0),
                source_set=sources,
                direct_regex_hit=direct,
                expected_code_match=code_match,
                expansion_type=None,
            )
        )
    return candidates
```

- [ ] **Step 3: Add `cheap_expand()` to `selector.py`**

Append to `src/retrieval/selector.py`:

```python
def cheap_expand(
    candidates: list[Candidate],
    source_rankings: dict[str, list[str]],
    rrf_scores: dict[str, float],
    query_info,                              # RewriteResult or None
    *,
    rescue_top_k: int = 10,
    seed_score_threshold: float = 0.5,
) -> list[Candidate]:
    """Mark same_parent / same_code candidates and add source diversity rescue.

    same_parent / same_code: flags are set on existing pool candidates (no new
    candidates added — they are already in the reranker top-100 pool).

    source diversity rescue: candidates from source_rankings[:rescue_top_k] that
    are NOT already in the pool are added with rerank_score=0.0.
    """
    existing: dict[str, Candidate] = {c.citation: c for c in candidates}
    source_sets = {src: set(ranking) for src, ranking in source_rankings.items()}
    extracted_set: set[str] = source_sets.get("extracted", set())
    expected_codes: set[str] = (
        set(query_info.expected_codes) if query_info is not None else set()
    )

    # ── Collect seed parents and codes ────────────────────────────────────
    seeds = [c for c in candidates if _is_seed(c, seed_score_threshold)]
    seed_citations: set[str] = {c.citation for c in seeds}
    seed_parents: set[str] = set()
    seed_codes: set[str] = set()
    for c in seeds:
        p = _extract_bge_parent(c.citation)
        if p:
            seed_parents.add(p)
        code = _extract_law_code(c.citation)
        if code:
            seed_codes.add(code)

    # ── Mark same_parent / same_code on non-seed pool candidates ──────────
    for c in candidates:
        if c.citation in seed_citations:
            continue
        p = _extract_bge_parent(c.citation)
        if p and p in seed_parents:
            c.expansion_type = "same_parent"
            continue
        code = _extract_law_code(c.citation)
        if code and code in seed_codes:
            c.expansion_type = "same_code"

    # ── Source diversity rescue ───────────────────────────────────────────
    new_candidates: list[Candidate] = []
    for src, ranking in source_rankings.items():
        for citation in ranking[:rescue_top_k]:
            if citation in existing:
                continue
            sources = frozenset(s for s, ss in source_sets.items() if citation in ss)
            direct = citation in extracted_set
            code = _extract_law_code(citation)
            code_match = code is not None and code in expected_codes
            new_candidates.append(
                Candidate(
                    citation=citation,
                    rerank_score=0.0,
                    rrf_score=rrf_scores.get(citation, 0.0),
                    source_set=sources,
                    direct_regex_hit=direct,
                    expected_code_match=code_match,
                    expansion_type=f"{src}_rescue",
                )
            )
            existing[citation] = new_candidates[-1]  # prevent duplicate rescue

    return candidates + new_candidates
```

- [ ] **Step 4: Commit**

```bash
git add src/retrieval/selector.py
git commit -m "selector: add Candidate dataclass, build_candidates, cheap_expand"
```

---

## Task 4: Add `llm_verify()` to `selector.py`

**Files:**
- Modify: `src/retrieval/selector.py`

- [ ] **Step 1: Add LLM verifier system prompt and helpers**

Append to `src/retrieval/selector.py` (after cheap_expand):

```python
_VERIFIER_SYSTEM = """\
You are a Swiss legal citation relevance assessor.

Given a legal question and a list of candidate citations retrieved from Swiss legal \
databases, assess each candidate's relevance to the question.

For each candidate assign a relevance score:
  3 = directly answers the legal issue
  2 = useful supporting authority
  1 = weak or background relevance
  0 = unrelated or misleading

Provide a brief reason (1 sentence in English).

Also provide estimated_answer_count: how many total citations would be needed to \
fully answer the legal question. Simple questions need fewer citations (e.g. 3-5), \
questions with multiple independent legal sub-issues need more (e.g. 10-20).

STRICT RULES:
- You MUST score ALL candidates — the output list must have exactly the same count as input
- You MUST use ONLY the id values provided — never generate new ids
- estimated_answer_count must be a positive integer

Output exactly one valid JSON object matching this schema:
{
  "candidate_scores": [{"id": "...", "relevance": 0|1|2|3, "reason": "..."}],
  "estimated_answer_count": <integer>
}
"""


def _build_verifier_user_msg(
    query: str,
    query_info,
    items: list[dict],
) -> str:
    ctx: dict = {"question": query, "candidates": items}
    if query_info is not None:
        ctx["legal_issue"] = query_info.legal_issue
        ctx["expected_codes"] = query_info.expected_codes
        ctx["expected_articles"] = query_info.expected_articles
    return json.dumps(ctx, ensure_ascii=False)


def _apply_verifier_response(
    data: dict,
    id_to_candidate: dict[str, "Candidate"],
) -> int:
    """Parse LLM response, fill relevance into candidates. Returns n_LLM."""
    for item in data.get("candidate_scores", []):
        cid = item.get("id")
        if cid not in id_to_candidate:
            continue
        relevance = item.get("relevance")
        if not isinstance(relevance, int) or relevance not in (0, 1, 2, 3):
            continue
        c = id_to_candidate[cid]
        c.relevance = relevance
        c.llm_reason = str(item.get("reason", ""))

    raw = data.get("estimated_answer_count", 10)
    if not isinstance(raw, int) or raw < 1:
        return 10
    return min(int(raw), 100)
```

- [ ] **Step 2: Add `llm_verify()` function**

Append to `src/retrieval/selector.py`:

```python
def llm_verify(
    candidates: list[Candidate],
    query: str,
    query_info,
    corpus_texts: dict[str, str],
    *,
    verifier_top_k: int = 60,
    text_snippet_len: int = 300,
) -> tuple[list[Candidate], int]:
    """Score candidates with LLM. Returns (candidates, n_LLM).

    Fills relevance / llm_reason on the top verifier_top_k candidates (by
    rerank_score). Remaining candidates keep relevance=None.

    On LLM failure: retries once, then returns n_LLM=10 with all relevance=None.
    """
    import os
    import sys as _sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from llm.client import chat_json

    top = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)[:verifier_top_k]
    id_to_candidate: dict[str, "Candidate"] = {f"C{i:03d}": c for i, c in enumerate(top)}

    items = [
        {
            "id": cid,
            "citation": c.citation,
            "text": (corpus_texts.get(c.citation) or "")[:text_snippet_len],
            "rerank_score": round(c.rerank_score, 4),
            "direct_hit": c.direct_regex_hit,
        }
        for cid, c in id_to_candidate.items()
    ]
    user_msg = _build_verifier_user_msg(query, query_info, items)

    for attempt in range(2):
        try:
            data = chat_json(_VERIFIER_SYSTEM, user_msg)
            n_llm = _apply_verifier_response(data, id_to_candidate)
            return candidates, n_llm
        except Exception as exc:
            print(f"[llm_verify] attempt {attempt + 1} failed: {exc}", file=sys.stderr)

    # Both attempts failed — return with all relevance=None
    print("[llm_verify] both attempts failed; using n_LLM=10", file=sys.stderr)
    return candidates, 10
```

- [ ] **Step 3: Commit**

```bash
git add src/retrieval/selector.py
git commit -m "selector: add llm_verify with retry and graceful fallback"
```

---

## Task 5: Add `fuse_scores`, `adaptive_count`, `assemble`, `run_selector`

**Files:**
- Modify: `src/retrieval/selector.py`

- [ ] **Step 1: Add `fuse_scores()`**

Append to `src/retrieval/selector.py`:

```python
_LLM_BOOST: dict[int, float] = {0: -0.40, 1: -0.10, 2: 0.15, 3: 0.35}


def fuse_scores(candidates: list[Candidate]) -> list[Candidate]:
    """Compute final_score for each candidate; return sorted descending.

    final_score = rerank_score + llm_relevance_boost + rule_boost
    """
    for c in candidates:
        llm_b = _LLM_BOOST.get(c.relevance, 0.0) if c.relevance is not None else 0.0

        rule_b = 0.0
        if c.direct_regex_hit:
            rule_b += 0.25
        if c.expected_code_match:
            rule_b += 0.06
        if len(c.source_set) >= 2:
            rule_b += 0.04
        if c.expansion_type == "same_parent":
            rule_b += 0.03
        elif c.expansion_type == "same_code":
            rule_b += 0.02
        if c.expansion_type is not None and c.expansion_type.endswith("_rescue"):
            rule_b -= 0.05
        if c.direct_regex_hit and c.expansion_type is not None:
            rule_b -= 0.05  # direct hit but low in reranker pool — partially offset

        c.final_score = c.rerank_score + llm_b + rule_b

    candidates.sort(key=lambda c: c.final_score, reverse=True)
    return candidates
```

- [ ] **Step 2: Add `adaptive_count()` and `_elbow()`**

Append to `src/retrieval/selector.py`:

```python
def _elbow(scores: list[float]) -> int:
    """Return elbow point count using Kneedle algorithm (distance to min-max line)."""
    if len(scores) <= 2:
        return len(scores)
    from kneed import KneeLocator

    x = list(range(len(scores)))
    knl = KneeLocator(x, scores, curve="convex", direction="decreasing")
    knee = knl.knee
    return (knee + 1) if knee is not None else len(scores)


def adaptive_count(
    candidates: list[Candidate],
    n_llm: int,
    *,
    min_val: int = 3,
    max_val: int = 40,
) -> int:
    """Compute n_final = clamp(round(4/7 * n_LLM + 3/7 * n_elbow), min, max).

    candidates must already be sorted by final_score descending (fuse_scores output).
    """
    scores = [c.final_score for c in candidates]
    n_elbow = _elbow(scores)
    raw = round(4 / 7 * n_llm + 3 / 7 * n_elbow)
    return max(min_val, min(max_val, raw))
```

- [ ] **Step 3: Add `assemble()`**

Append to `src/retrieval/selector.py`:

```python
def assemble(
    candidates: list[Candidate],
    n_final: int,
    *,
    max_per_parent: int = 3,
    rescue_score_floor: float = 0.1,
) -> list[str]:
    """Select final citation list from scored candidates.

    Steps (in order):
    1. Remove low-confidence rescue candidates (rerank_score < rescue_score_floor)
    2. Deduplicate (preserve final_score order)
    3. Take top n_final with BGE parent constraint (max_per_parent per BGE case)
    4. Append any direct_regex_hit candidates beyond n_final cutoff

    All output citations come from Candidate.citation — no new citations are generated.
    """
    # Step 1: filter low-quality rescue
    filtered = [
        c for c in candidates
        if not (
            c.expansion_type is not None
            and c.expansion_type.endswith("_rescue")
            and c.rerank_score < rescue_score_floor
        )
    ]

    # Step 2: dedup (keep highest final_score occurrence = first in sorted list)
    seen: set[str] = set()
    deduped: list[Candidate] = []
    for c in filtered:
        if c.citation not in seen:
            seen.add(c.citation)
            deduped.append(c)

    # Step 3 & 4: select with parent constraint; collect direct_hit overflow
    parent_counts: dict[str, int] = {}
    result: list[str] = []
    direct_hit_extras: list[str] = []

    for c in deduped:
        parent = _extract_bge_parent(c.citation)
        if parent and parent_counts.get(parent, 0) >= max_per_parent:
            continue

        if len(result) < n_final:
            result.append(c.citation)
            if parent:
                parent_counts[parent] = parent_counts.get(parent, 0) + 1
        elif c.direct_regex_hit:
            direct_hit_extras.append(c.citation)
            if parent:
                parent_counts[parent] = parent_counts.get(parent, 0) + 1

    return result + direct_hit_extras
```

- [ ] **Step 4: Add `run_selector()` entry point**

Append to `src/retrieval/selector.py`:

```python
def run_selector(
    query: str,
    query_info,                              # RewriteResult or None
    scored: list[tuple[str, float]],         # rerank_with_scores() output
    rrf_scores: dict[str, float],            # weighted_rrf() score dict
    source_rankings: dict[str, list[str]],   # per-path citation lists
    corpus_texts: dict[str, str],            # citation -> indexed_text
    *,
    # cheap_expand params
    rescue_top_k: int = 10,
    seed_score_threshold: float = 0.5,
    # llm_verify params
    use_llm_verify: bool = True,
    verifier_top_k: int = 60,
    text_snippet_len: int = 300,
    # adaptive_count params
    min_citations: int = 3,
    max_citations: int = 40,
    # assemble params
    max_per_parent: int = 3,
    rescue_score_floor: float = 0.1,
) -> list[str]:
    """Run the full selector pipeline and return the final citation list."""
    candidates = build_candidates(scored, rrf_scores, source_rankings, query_info)
    candidates = cheap_expand(
        candidates, source_rankings, rrf_scores, query_info,
        rescue_top_k=rescue_top_k,
        seed_score_threshold=seed_score_threshold,
    )

    n_llm = 10
    if use_llm_verify:
        candidates, n_llm = llm_verify(
            candidates, query, query_info, corpus_texts,
            verifier_top_k=verifier_top_k,
            text_snippet_len=text_snippet_len,
        )

    candidates = fuse_scores(candidates)
    n_final = adaptive_count(
        candidates, n_llm, min_val=min_citations, max_val=max_citations
    )
    return assemble(
        candidates, n_final,
        max_per_parent=max_per_parent,
        rescue_score_floor=rescue_score_floor,
    )
```

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/selector.py
git commit -m "selector: add fuse_scores, adaptive_count, assemble, run_selector"
```

---

## Task 6: Tests for `selector.py`

**Files:**
- Create: `tests/retrieval/test_selector.py`

- [ ] **Step 1: Create test file**

Create `tests/retrieval/test_selector.py`:

```python
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import retrieval.selector as sel
from retrieval.selector import (
    Candidate,
    adaptive_count,
    assemble,
    build_candidates,
    cheap_expand,
    fuse_scores,
    llm_verify,
    run_selector,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_candidate(
    citation: str,
    rerank_score: float = 0.5,
    rrf_score: float = 0.1,
    source_set=None,
    direct_regex_hit: bool = False,
    expected_code_match: bool = False,
    expansion_type=None,
    relevance=None,
) -> Candidate:
    return Candidate(
        citation=citation,
        rerank_score=rerank_score,
        rrf_score=rrf_score,
        source_set=frozenset(source_set or ["bm25_court"]),
        direct_regex_hit=direct_regex_hit,
        expected_code_match=expected_code_match,
        expansion_type=expansion_type,
        relevance=relevance,
    )


CORPUS_TEXTS = {
    "BGE 145 IV 154 E. 1.1": "Bundesgericht entschied über Betrug",
    "Art. 146 StGB": "Wer in der Absicht, sich zu bereichern",
    "Art. 221 StPO": "Die Untersuchungshaft",
}

# ── build_candidates ──────────────────────────────────────────────────────────

class FakeQueryInfo:
    expected_codes = ["StGB", "OR"]
    expected_articles = []
    legal_issue = "Betrug"
    search_terms = {"de": [], "fr": []}


def test_build_candidates_fills_source_set():
    scored = [("Art. 146 StGB", 0.9), ("BGE 145 IV 154 E. 1.1", 0.7)]
    rrf_scores = {"Art. 146 StGB": 0.05, "BGE 145 IV 154 E. 1.1": 0.04}
    source_rankings = {
        "bm25_law": ["Art. 146 StGB", "BGE 145 IV 154 E. 1.1"],
        "dense_court": ["BGE 145 IV 154 E. 1.1"],
    }
    candidates = build_candidates(scored, rrf_scores, source_rankings, FakeQueryInfo())
    art = next(c for c in candidates if "StGB" in c.citation and "Art" in c.citation)
    bge = next(c for c in candidates if "BGE" in c.citation)
    assert "bm25_law" in art.source_set
    assert "dense_court" not in art.source_set
    assert "dense_court" in bge.source_set
    assert "bm25_law" in bge.source_set


def test_build_candidates_direct_regex_hit():
    scored = [("Art. 146 StGB", 0.8)]
    rrf_scores = {"Art. 146 StGB": 0.05}
    source_rankings = {
        "extracted": ["Art. 146 StGB"],
        "bm25_law": ["Art. 146 StGB"],
    }
    candidates = build_candidates(scored, rrf_scores, source_rankings, None)
    assert candidates[0].direct_regex_hit is True


def test_build_candidates_expected_code_match():
    scored = [("Art. 146 StGB", 0.8), ("BGE 145 IV 154 E. 1.1", 0.6)]
    rrf_scores = {"Art. 146 StGB": 0.05, "BGE 145 IV 154 E. 1.1": 0.04}
    source_rankings = {"bm25_law": ["Art. 146 StGB", "BGE 145 IV 154 E. 1.1"]}
    candidates = build_candidates(scored, rrf_scores, source_rankings, FakeQueryInfo())
    art = next(c for c in candidates if "Art. 146" in c.citation)
    bge = next(c for c in candidates if "BGE" in c.citation)
    assert art.expected_code_match is True    # StGB in expected_codes
    assert bge.expected_code_match is False   # BGE has no law code


def test_build_candidates_no_query_info():
    scored = [("Art. 146 StGB", 0.8)]
    rrf_scores = {"Art. 146 StGB": 0.05}
    candidates = build_candidates(scored, rrf_scores, {"bm25_law": ["Art. 146 StGB"]}, None)
    assert candidates[0].expected_code_match is False
    assert candidates[0].direct_regex_hit is False

# ── cheap_expand ──────────────────────────────────────────────────────────────

def test_cheap_expand_marks_same_parent():
    """High-score BGE seed → same-parent candidate in pool gets marked."""
    c_seed = _make_candidate("BGE 145 IV 154 E. 1.1", rerank_score=0.8)
    c_sibling = _make_candidate("BGE 145 IV 154 E. 2.3", rerank_score=0.3)
    c_other = _make_candidate("Art. 146 StGB", rerank_score=0.6)
    candidates = [c_seed, c_sibling, c_other]
    result = cheap_expand(candidates, {}, {}, None)
    sibling = next(c for c in result if c.citation == "BGE 145 IV 154 E. 2.3")
    assert sibling.expansion_type == "same_parent"
    # seed itself should NOT be marked
    seed = next(c for c in result if c.citation == "BGE 145 IV 154 E. 1.1")
    assert seed.expansion_type is None


def test_cheap_expand_marks_same_code():
    """High-score StPO seed → other StPO candidate in pool gets marked."""
    c_seed = _make_candidate("Art. 221 StPO", rerank_score=0.75)
    c_same_code = _make_candidate("Art. 237 StPO", rerank_score=0.2)
    candidates = [c_seed, c_same_code]
    result = cheap_expand(candidates, {}, {}, None)
    marked = next(c for c in result if c.citation == "Art. 237 StPO")
    assert marked.expansion_type == "same_code"


def test_cheap_expand_adds_rescue_candidates():
    """Citations in source_rankings[:rescue_top_k] but not in pool get added."""
    c_existing = _make_candidate("Art. 146 StGB", rerank_score=0.8)
    candidates = [c_existing]
    source_rankings = {
        "bm25_law": ["Art. 146 StGB", "Art. 147 StGB"],  # Art. 147 is NOT in pool
    }
    rrf_scores = {"Art. 146 StGB": 0.05, "Art. 147 StGB": 0.03}
    result = cheap_expand(candidates, source_rankings, rrf_scores, None, rescue_top_k=5)
    citations = [c.citation for c in result]
    assert "Art. 147 StGB" in citations
    rescue = next(c for c in result if c.citation == "Art. 147 StGB")
    assert rescue.expansion_type == "bm25_law_rescue"
    assert rescue.rerank_score == 0.0
    assert rescue.rrf_score == pytest.approx(0.03)


def test_cheap_expand_does_not_duplicate_rescue():
    """Candidate already in pool is not added again as rescue."""
    c = _make_candidate("Art. 146 StGB", rerank_score=0.8)
    source_rankings = {"bm25_law": ["Art. 146 StGB"]}
    result = cheap_expand([c], source_rankings, {}, None, rescue_top_k=5)
    assert sum(1 for x in result if x.citation == "Art. 146 StGB") == 1

# ── fuse_scores ───────────────────────────────────────────────────────────────

def test_fuse_scores_direct_regex_hit_boost():
    c = _make_candidate("Art. 146 StGB", rerank_score=0.5, direct_regex_hit=True)
    fuse_scores([c])
    assert c.final_score == pytest.approx(0.5 + 0.25, abs=1e-6)


def test_fuse_scores_llm_relevance_boost():
    c = _make_candidate("Art. 146 StGB", rerank_score=0.5, relevance=3)
    fuse_scores([c])
    assert c.final_score == pytest.approx(0.5 + 0.35, abs=1e-6)


def test_fuse_scores_rescue_penalty():
    c = _make_candidate("Art. 147 StGB", rerank_score=0.0, expansion_type="bm25_law_rescue")
    fuse_scores([c])
    assert c.final_score == pytest.approx(0.0 - 0.05, abs=1e-6)


def test_fuse_scores_direct_hit_plus_rescue_penalty():
    """direct_regex_hit AND expansion_type is not None → -0.05 extra."""
    c = _make_candidate(
        "Art. 146 StGB", rerank_score=0.0,
        direct_regex_hit=True, expansion_type="extracted_rescue"
    )
    fuse_scores([c])
    # +0.25 (direct) -0.05 (rescue) -0.05 (direct+expansion) = +0.15
    assert c.final_score == pytest.approx(0.0 + 0.25 - 0.05 - 0.05, abs=1e-6)


def test_fuse_scores_sorted_descending():
    candidates = [
        _make_candidate("A", rerank_score=0.3),
        _make_candidate("B", rerank_score=0.8),
        _make_candidate("C", rerank_score=0.1),
    ]
    result = fuse_scores(candidates)
    scores = [c.final_score for c in result]
    assert scores == sorted(scores, reverse=True)

# ── adaptive_count ────────────────────────────────────────────────────────────

def test_adaptive_count_respects_min():
    candidates = [_make_candidate(f"C{i}", rerank_score=0.9 - i * 0.1) for i in range(2)]
    for c in candidates:
        c.final_score = c.rerank_score
    assert adaptive_count(candidates, n_llm=1, min_val=3, max_val=40) >= 3


def test_adaptive_count_respects_max():
    candidates = [_make_candidate(f"C{i}", rerank_score=0.9) for i in range(100)]
    for c in candidates:
        c.final_score = 0.9
    assert adaptive_count(candidates, n_llm=100, min_val=3, max_val=40) <= 40


def test_adaptive_count_formula():
    """n_final = clamp(round(4/7 * n_llm + 3/7 * n_elbow), 3, 40)."""
    # Flat scores → KneeLocator returns None → n_elbow = len(candidates)
    # Use 10 candidates with identical scores so elbow = 10
    # Then manually inject known values via mocking _elbow
    candidates = [_make_candidate(f"C{i}") for i in range(5)]
    for c in candidates:
        c.final_score = 0.5
    with patch.object(sel, "_elbow", return_value=8):
        n = adaptive_count(candidates, n_llm=7, min_val=3, max_val=40)
    expected = round(4 / 7 * 7 + 3 / 7 * 8)  # round(4 + 24/7) = round(7.43) = 7
    assert n == max(3, min(40, expected))

# ── assemble ──────────────────────────────────────────────────────────────────

def test_assemble_top_n():
    candidates = [_make_candidate(f"C{i}") for i in range(10)]
    for i, c in enumerate(candidates):
        c.final_score = 1.0 - i * 0.1
    result = assemble(candidates, n_final=3)
    assert result == ["C0", "C1", "C2"]


def test_assemble_parent_constraint():
    """Same BGE parent: at most max_per_parent=2 citations kept."""
    cs = [
        _make_candidate("BGE 145 IV 154 E. 1.1", rerank_score=0.9),
        _make_candidate("BGE 145 IV 154 E. 2.0", rerank_score=0.8),
        _make_candidate("BGE 145 IV 154 E. 3.0", rerank_score=0.7),
        _make_candidate("Art. 146 StGB", rerank_score=0.6),
    ]
    for i, c in enumerate(cs):
        c.final_score = 1.0 - i * 0.1
    result = assemble(cs, n_final=4, max_per_parent=2)
    bge_count = sum(1 for r in result if "BGE 145 IV 154" in r)
    assert bge_count == 2
    assert "Art. 146 StGB" in result


def test_assemble_direct_hit_overflow():
    """direct_regex_hit candidate beyond n_final is appended."""
    cs = [
        _make_candidate("Art. 146 StGB", rerank_score=0.9),
        _make_candidate("Art. 221 StPO", rerank_score=0.8),
        _make_candidate("Art. 237 StPO", rerank_score=0.7, direct_regex_hit=True),
    ]
    for i, c in enumerate(cs):
        c.final_score = 1.0 - i * 0.1
    result = assemble(cs, n_final=2)
    assert result[:2] == ["Art. 146 StGB", "Art. 221 StPO"]
    assert "Art. 237 StPO" in result   # forced in via direct_regex_hit


def test_assemble_rescue_floor_removes_low_quality():
    """Rescue candidate with rerank_score < floor is excluded."""
    cs = [
        _make_candidate("Art. 146 StGB", rerank_score=0.9),
        _make_candidate("Art. 999 ZGB", rerank_score=0.05, expansion_type="bm25_court_rescue"),
    ]
    for i, c in enumerate(cs):
        c.final_score = 1.0 - i * 0.1
    result = assemble(cs, n_final=5, rescue_score_floor=0.1)
    assert "Art. 999 ZGB" not in result


def test_assemble_dedup():
    """Duplicate citations appear only once."""
    c1 = _make_candidate("Art. 146 StGB", rerank_score=0.9)
    c2 = _make_candidate("Art. 146 StGB", rerank_score=0.5)
    c1.final_score = 0.9
    c2.final_score = 0.5
    result = assemble([c1, c2], n_final=5)
    assert result.count("Art. 146 StGB") == 1

# ── llm_verify ────────────────────────────────────────────────────────────────

def test_llm_verify_fills_relevance():
    c = _make_candidate("Art. 146 StGB", rerank_score=0.9)
    fake_response = {
        "candidate_scores": [{"id": "C000", "relevance": 3, "reason": "direct"}],
        "estimated_answer_count": 7,
    }
    # chat_json is imported at module level in selector.py → patch via selector module
    with patch("retrieval.selector.chat_json", return_value=fake_response):
        result_candidates, n_llm = llm_verify(
            [c], "fraud query", FakeQueryInfo(), CORPUS_TEXTS
        )
    assert result_candidates[0].relevance == 3
    assert result_candidates[0].llm_reason == "direct"
    assert n_llm == 7


def test_llm_verify_fallback_on_failure():
    """Two failures → relevance stays None, n_LLM = 10."""
    c = _make_candidate("Art. 146 StGB", rerank_score=0.9)
    with patch("retrieval.selector.chat_json", side_effect=RuntimeError("API error")):
        result_candidates, n_llm = llm_verify(
            [c], "query", FakeQueryInfo(), CORPUS_TEXTS
        )
    assert result_candidates[0].relevance is None
    assert n_llm == 10
```

- [ ] **Step 2: Run tests — expect PASS**

```bash
conda run -n agent pytest tests/retrieval/test_selector.py -v 2>&1 | tail -30
```

Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/retrieval/test_selector.py
git commit -m "tests: add comprehensive selector pipeline unit tests"
```

---

## Task 7: Integrate selector into `run.py`

**Files:**
- Modify: `src/query/run.py`

- [ ] **Step 1: Update imports and module-level flags**

Replace the imports block and `_USE_*` flags at the top of `run.py` (lines 1–36):

```python
"""
Run citation prediction pipeline on a query CSV and write predictions.

Usage:
    conda run -n agent python src/query/run.py
    conda run -n agent python src/query/run.py --input dataset/test.csv --output results/test_predictions.csv
"""

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from retrieval.rerank import _load_reranker, rerank, rerank_with_scores, reranker_exists
from retrieval.rrf import weighted_rrf
from retrieval.selector import run_selector

ROOT_DIR    = os.path.join(os.path.dirname(__file__), "../..")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
DEFAULT_INPUT  = os.path.join(DATASET_DIR, "val.csv")
DEFAULT_OUTPUT = os.path.join(ROOT_DIR, "results/predictions.csv")
DEFAULT_REWRITE_LOG_DIR = os.path.join(ROOT_DIR, "results/rewrite_logs")
_USE_DENSE_COURT: bool = dense_court_exists()
_USE_DENSE_LAW:   bool = dense_law_exists()
_USE_RERANK: bool = reranker_exists()
```

- [ ] **Step 2: Update `_print_run_config()` to include selector params**

Replace the `_print_run_config` function:

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
    use_select: bool,
    use_llm_verify: bool,
    verifier_top_k: int,
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
    print(f"  selector                = {use_select} (llm_verify={use_llm_verify}, verifier_top_k={verifier_top_k})", file=sys.stderr)
    print(f"  input (actual)          = {input_path}", file=sys.stderr)
    print(f"  output (actual)         = {output_path}", file=sys.stderr)
    print(f"  use_rewrite             = {use_rewrite}", file=sys.stderr)
    print(f"  rewrite_log_dir         = {rewrite_log_dir}", file=sys.stderr)
```

- [ ] **Step 3: Replace `predict_citations()` with updated pipeline**

Replace the entire `predict_citations` function:

```python
def predict_citations(
    query: str,
    *,
    use_rewrite: bool = True,
    use_rerank: bool = True,
    rerank_top_k: int = 100,
    rerank_batch_size: int = 32,
    use_select: bool = True,
    use_llm_verify: bool = True,
    verifier_top_k: int = 60,
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
    """Query pipeline: optional LLM rewrite + 5-way weighted RRF + optional reranker + selector."""
    if weight_court is not None:
        weight_bm25_court = weight_court
    if weight_law is not None:
        weight_bm25_law = weight_law

    search_text = None
    rewrite_result = None
    llm_articles: list[str] = []
    if use_rewrite:
        rewrite_result = rewrite_query(query)
        search_text = format_search_text(rewrite_result, lang="de")
        llm_articles = rewrite_result.expected_articles
        if llm_articles:
            search_text = search_text + " " + " ".join(llm_articles)
        if rewrite_log_dir and query_id:
            _log_rewrite(rewrite_log_dir, query_id, query, rewrite_result, search_text)

    extracted, bm25_court, bm25_law = retrieve_bm25_parts(
        query, search_text, k_court, k_law,
        extra_citations=llm_articles or None,
    )

    if llm_articles:
        seen: set[str] = set(llm_articles)
        merged = list(llm_articles)
        merged.extend(c for c in extracted if c not in seen)
        extracted = merged

    rankings: list[tuple[list[str], float]] = [
        (extracted,   weight_extracted),
        (bm25_court,  weight_bm25_court),
        (bm25_law,    weight_bm25_law),
    ]

    dense_court: list[str] = []
    dense_law: list[str] = []
    if _USE_DENSE_COURT or _USE_DENSE_LAW:
        dense_query = search_text if search_text is not None else query
        dense_court, dense_law = retrieve_dense_parts(
            dense_query,
            k_court=k_court,
            k_law=k_law,
            use_court=_USE_DENSE_COURT,
            use_law=_USE_DENSE_LAW,
        )
        if _USE_DENSE_COURT and dense_court:
            rankings.append((dense_court, weight_dense_court))
        if _USE_DENSE_LAW and dense_law:
            rankings.append((dense_law, weight_dense_law))

    rrf_result, rrf_scores = weighted_rrf(rankings, rrf_k=rrf_k)

    if use_rerank and _USE_RERANK:
        rerank_query = query
        if search_text is not None:
            rerank_query = query + " " + search_text
        scored = rerank_with_scores(
            rerank_query,
            rrf_result,
            corpus.get_corpus_texts(),
            top_k=rerank_top_k,
            batch_size=rerank_batch_size,
        )
    else:
        # No reranker: use rrf_scores as proxy, convert to scored list
        scored = [(cit, rrf_scores.get(cit, 0.0)) for cit in rrf_result[:rerank_top_k]]

    if use_select:
        source_rankings: dict[str, list[str]] = {
            "extracted": extracted,
            "bm25_court": bm25_court,
            "bm25_law": bm25_law,
        }
        if _USE_DENSE_COURT and dense_court:
            source_rankings["dense_court"] = dense_court
        if _USE_DENSE_LAW and dense_law:
            source_rankings["dense_law"] = dense_law

        return run_selector(
            query,
            rewrite_result,
            scored,
            rrf_scores,
            source_rankings,
            corpus.get_corpus_texts(),
            use_llm_verify=use_llm_verify,
            verifier_top_k=verifier_top_k,
        )

    # Fallback: no selector — return flat reranked list truncated to k
    return [cit for cit, _ in scored][:k]
```

- [ ] **Step 4: Update `_process_query()` to pass new params**

Replace `_process_query`:

```python
def _process_query(
    row: dict[str, str],
    *,
    use_rewrite: bool,
    use_rerank: bool,
    rerank_top_k: int,
    rerank_batch_size: int,
    use_select: bool,
    use_llm_verify: bool,
    verifier_top_k: int,
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
        use_select=use_select,
        use_llm_verify=use_llm_verify,
        verifier_top_k=verifier_top_k,
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

- [ ] **Step 5: Update `run()` to pass and log new params**

Replace the `run` function signature and body:

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
    use_select: bool = True,
    use_llm_verify: bool = True,
    verifier_top_k: int = 60,
    rewrite_log_dir: str | None = None,
    workers: int = 4,
) -> None:
    _print_run_config(
        input_path,
        output_path,
        use_rewrite=use_rewrite,
        rewrite_log_dir=rewrite_log_dir,
        use_rerank=use_rerank,
        rerank_top_k=rerank_top_k,
        rerank_batch_size=rerank_batch_size,
        use_select=use_select,
        use_llm_verify=use_llm_verify,
        verifier_top_k=verifier_top_k,
    )
    queries = load_queries(input_path)
    if not queries:
        write_predictions(output_path, [])
        print(f"Wrote 0 predictions → {output_path}", file=sys.stderr)
        return

    corpus.load_corpus()
    _load_bm25_index()
    if _USE_DENSE_COURT or _USE_DENSE_LAW:
        _load_dense_index(use_court=_USE_DENSE_COURT, use_law=_USE_DENSE_LAW)
    if use_rerank and _USE_RERANK:
        _load_reranker()

    process_kwargs = {
        "use_rewrite": use_rewrite,
        "use_rerank": use_rerank,
        "rerank_top_k": rerank_top_k,
        "rerank_batch_size": rerank_batch_size,
        "use_select": use_select,
        "use_llm_verify": use_llm_verify,
        "verifier_top_k": verifier_top_k,
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

    results: dict[int, tuple[str, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_process_query, row, **process_kwargs): i
            for i, row in enumerate(queries)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            qid, _ = results[i]
            print(f"[{done}/{len(queries)}] {qid}", file=sys.stderr)

    write_predictions(
        output_path, [results[i] for i in range(len(queries))]
    )
    print(f"Wrote {len(results)} predictions → {output_path}", file=sys.stderr)
    if use_rewrite and rewrite_log_dir:
        print(f"Rewrite logs → {rewrite_log_dir}", file=sys.stderr)
```

- [ ] **Step 6: Add new CLI arguments to `main()`**

In the `main()` function, add these three argument definitions after the existing `--rerank-batch-size` argument:

```python
    parser.add_argument(
        "--no-select",
        action="store_true",
        help="Disable selector pipeline; fall back to plain reranked[:k] output",
    )
    parser.add_argument(
        "--no-llm-verify",
        action="store_true",
        help="Run selector without LLM verifier (cheap expand + score fusion only)",
    )
    parser.add_argument(
        "--verifier-top-k",
        type=int,
        default=60,
        help="Number of candidates shown to LLM verifier (default: 60)",
    )
```

And update the `run(...)` call at the end of `main()` to pass the new params:

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
        use_select=not args.no_select,
        use_llm_verify=not args.no_llm_verify,
        verifier_top_k=args.verifier_top_k,
        rewrite_log_dir=None if args.no_rewrite_log else args.rewrite_log,
        workers=args.workers,
    )
```

- [ ] **Step 7: Smoke-test with `--no-select` to verify no regressions**

```bash
conda run -n agent python src/query/run.py --help 2>&1 | grep -E "select|verify"
```

Expected output contains:
```
  --no-select           Disable selector pipeline; fall back to plain reranked[:k] output
  --no-llm-verify       Run selector without LLM verifier (cheap expand + score fusion only)
  --verifier-top-k      Number of candidates shown to LLM verifier (default: 60)
```

- [ ] **Step 8: Run full test suite**

```bash
conda run -n agent pytest tests/ -v 2>&1 | tail -20
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/query/run.py
git commit -m "run: integrate selector pipeline; add --no-select, --no-llm-verify, --verifier-top-k"
```
