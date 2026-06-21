# Adaptive Count n_rel23 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `n_rel23` (count of candidates with LLM relevance 2 or 3) as a third weighted term in `adaptive_count()`, using weights 0.2 / 0.4 / 0.4 for `n_LLM` / `n_elbow` / `n_rel23`.

**Architecture:** Single-function change inside `adaptive_count()` in `src/retrieval/selector.py`. Count `n_rel23` from the `candidates` list already passed in; no signature or call-site changes. Tests updated first (TDD), then implementation.

**Tech Stack:** Python 3.11, pytest, `kneed` (unchanged — only used by `_elbow`)

**Spec:** `docs/superpowers/specs/2026-06-21-adaptive-count-rel23-design.md`

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/retrieval/selector.py:342-357` | `adaptive_count()` — add `n_rel23`, new formula, update docstring |
| Modify | `tests/retrieval/test_selector.py:213-224` | Update formula test; add `n_rel23` tests |
| Modify | `docs/superpowers/specs/2026-06-21-adaptive-count-rel23-design.md` | Mark status 已实现 after code lands |

**Out of scope:** `fuse_scores`, `llm_verify`, CLI, weight tuning, `run_selector` call chain.

**Note:** If `adaptive_count()` contains a debug `print(...)` or other local edits unrelated to this feature, remove the debug print when touching the function. Do not change `_elbow` input slicing unless required by this spec (spec uses full `scores` list).

---

## Task 1: Update and add failing tests

**Files:**
- Modify: `tests/retrieval/test_selector.py:213-224`

- [ ] **Step 1: Replace `test_adaptive_count_formula` and add two new tests**

In `tests/retrieval/test_selector.py`, replace `test_adaptive_count_formula` (lines 213–224) and append the two new tests immediately after it (before the `# ── assemble` section):

```python
def test_adaptive_count_formula():
    """n_final = clamp(round(0.2*n_llm + 0.4*n_elbow + 0.4*n_rel23), 3, 40)."""
    candidates = [_make_candidate(f"C{i}") for i in range(5)]
    for c in candidates:
        c.final_score = 0.5
    with patch.object(sel, "_elbow", return_value=8):
        n = adaptive_count(candidates, n_llm=7, min_val=3, max_val=40)
    # n_rel23=0 (all relevance=None)
    expected = round(0.2 * 7 + 0.4 * 8 + 0.4 * 0)  # round(4.6) = 5
    assert n == max(3, min(40, expected))


def test_adaptive_count_includes_rel23():
    """relevance 2/3 candidates contribute to n_rel23."""
    candidates = [
        _make_candidate(f"C{i}", relevance=3 if i < 5 else 0) for i in range(10)
    ]
    for c in candidates:
        c.final_score = 0.5
    with patch.object(sel, "_elbow", return_value=8):
        n = adaptive_count(candidates, n_llm=7, min_val=3, max_val=40)
    # n_rel23=5; expected = round(0.2*7 + 0.4*8 + 0.4*5) = round(6.6) = 7
    assert n == 7


def test_adaptive_count_rel2_counts():
    """relevance=2 counts; relevance=1 does not."""
    candidates = [
        _make_candidate(f"C{i}", relevance=2 if i < 3 else 1) for i in range(10)
    ]
    for c in candidates:
        c.final_score = 0.5
    with patch.object(sel, "_elbow", return_value=5):
        n = adaptive_count(candidates, n_llm=10, min_val=3, max_val=40)
    # n_rel23=3; expected = round(0.2*10 + 0.4*5 + 0.4*3) = round(4.2) = 4
    assert n == 4
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd /root/LawAgent && conda run -n agent pytest tests/retrieval/test_selector.py::test_adaptive_count_formula tests/retrieval/test_selector.py::test_adaptive_count_includes_rel23 tests/retrieval/test_selector.py::test_adaptive_count_rel2_counts -v
```

Expected: `test_adaptive_count_formula` and the two new tests FAIL (old formula still uses `4/7` and `3/7`, no `n_rel23`).

- [ ] **Step 3: Commit tests only**

```bash
cd /root/LawAgent && git add tests/retrieval/test_selector.py && git commit -m "$(cat <<'EOF'
test: update adaptive_count tests for n_rel23 weighted formula

EOF
)"
```

---

## Task 2: Implement new formula in `adaptive_count`

**Files:**
- Modify: `src/retrieval/selector.py:342-357`

- [ ] **Step 1: Update `adaptive_count()`**

Replace the function body and docstring:

```python
def adaptive_count(
    candidates: list[Candidate],
    n_llm: int,
    *,
    min_val: int = 3,
    max_val: int = 40,
) -> int:
    """Compute n_final = clamp(round(0.2*n_LLM + 0.4*n_elbow + 0.4*n_rel23), min, max).

    n_rel23 = count of candidates with relevance in (2, 3).
    candidates must already be sorted by final_score descending (fuse_scores output).
    """
    scores = [c.final_score for c in candidates]
    n_elbow = _elbow(scores)
    n_rel23 = sum(1 for c in candidates if c.relevance in (2, 3))
    raw = round(0.2 * n_llm + 0.4 * n_elbow + 0.4 * n_rel23)
    return max(min_val, min(max_val, raw))
```

Remove any debug `print(...)` that may exist in this function.

- [ ] **Step 2: Run all adaptive_count tests — expect PASS**

```bash
cd /root/LawAgent && conda run -n agent pytest tests/retrieval/test_selector.py -k adaptive_count -v
```

Expected: 5 passed (`test_adaptive_count_respects_min`, `test_adaptive_count_respects_max`, `test_adaptive_count_formula`, `test_adaptive_count_includes_rel23`, `test_adaptive_count_rel2_counts`).

- [ ] **Step 3: Run full selector test suite — expect PASS**

```bash
cd /root/LawAgent && conda run -n agent pytest tests/retrieval/test_selector.py -v
```

Expected: all tests in file PASS (no regressions in `assemble`, `fuse_scores`, `llm_verify`, etc.).

- [ ] **Step 4: Commit implementation**

```bash
cd /root/LawAgent && git add src/retrieval/selector.py && git commit -m "$(cat <<'EOF'
feat(selector): add n_rel23 term to adaptive citation count

Weight n_LLM, n_elbow, and high-relevance candidate count at 0.2/0.4/0.4.
EOF
)"
```

---

## Task 3: Mark spec as implemented

**Files:**
- Modify: `docs/superpowers/specs/2026-06-21-adaptive-count-rel23-design.md:4`

- [ ] **Step 1: Update spec status**

Change line 4 from `**状态：** 待实现` to `**状态：** 已实现`.

- [ ] **Step 2: Commit**

```bash
cd /root/LawAgent && git add docs/superpowers/specs/2026-06-21-adaptive-count-rel23-design.md && git commit -m "$(cat <<'EOF'
docs: mark adaptive-count-rel23 spec as implemented

EOF
)"
```

---

## Spec Coverage Checklist

| Spec requirement | Task |
|------------------|------|
| `n_rel23 = sum(relevance in (2,3))` | Task 2 |
| Formula `0.2/0.4/0.4` | Task 1 tests + Task 2 |
| Verifier skip → `n_rel23=0` (implicit) | Task 1 `test_adaptive_count_formula` |
| min/max clamp unchanged | Task 1 existing tests still pass |
| No call-site changes | Task 2 only touches `adaptive_count` |
| Docstring updated | Task 2 |

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-21-adaptive-count-rel23.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — dispatch a fresh subagent per task, review between tasks
2. **Inline Execution** — implement all tasks in this session with checkpoints

Which approach?
