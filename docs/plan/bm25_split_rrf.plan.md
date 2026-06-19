---
name: BM25 Split RRF
overview: 将 BM25 召回改为 court/law 双索引分路 top-k，query 正则提取作为第三路，加权 RRF 融合后输出 k 条候选。
todos:
  - id: rrf-module
    content: 新建 src/retrieval/rrf.py + tests/retrieval/test_rrf.py
    status: completed
  - id: dual-index
    content: 修改 build_bm25.py 构建 bm25_court/ 和 bm25_law/
    status: completed
  - id: bm25-retrieval
    content: 修改 src/retrieval/bm25.py 双索引三路召回 + RRF
    status: completed
  - id: query-cli
    content: 修改 src/query/run.py 暴露 k/权重 CLI 参数
    status: completed
  - id: verify
    content: 端到端 query → eval 验证
    status: completed
isProject: false
---

# BM25 分路召回 + 加权 RRF 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** court / law 分路 BM25 召回，query 提取作第三路，加权 RRF 融合，输出可配置条数。

**Architecture:** 双 BM25 索引（`bm25_court/`、`bm25_law/`）；`weighted_rrf()` 纯函数融合三路 ranking；`retrieve_bm25()` 编排召回与截断；CLI 透传参数。

**Tech Stack:** Python 3, bm25s, pytest

**设计文档:** [docs/superpowers/specs/2026-06-19-bm25-split-rrf-design.md](../superpowers/specs/2026-06-19-bm25-split-rrf-design.md)

---

## 文件结构

| 文件 | 职责 |
|------|------|
| `src/retrieval/rrf.py` | `weighted_rrf()` 纯函数 |
| `tests/retrieval/test_rrf.py` | RRF 单元测试 |
| `src/retrieval/bm25.py` | 双索引加载、三路召回、调 RRF |
| `src/indexing/build_bm25.py` | 按 source 构建两个索引 |
| `src/query/run.py` | CLI 参数扩展 |

---

### Task 1: weighted_rrf 模块

**Files:**
- Create: `src/retrieval/rrf.py`
- Create: `tests/retrieval/test_rrf.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/retrieval/test_rrf.py
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from retrieval.rrf import weighted_rrf


def test_single_ranking():
    rankings = [(["a", "b", "c"], 1.0)]
    assert weighted_rrf(rankings, rrf_k=60) == ["a", "b", "c"]


def test_multi_ranking_accumulates():
    rankings = [
        (["a", "b"], 1.0),
        (["b", "c"], 1.0),
    ]
    result = weighted_rrf(rankings, rrf_k=60)
    assert result[0] == "b"  # appears in both lists, highest RRF score
    assert set(result) == {"a", "b", "c"}


def test_weighted_ranking():
    rankings = [
        (["a"], 1.0),
        (["b"], 3.0),
    ]
    result = weighted_rrf(rankings, rrf_k=60)
    assert result[0] == "b"  # higher weight wins despite same rank


def test_empty_rankings():
    assert weighted_rrf([], rrf_k=60) == []


def test_empty_single_list():
    assert weighted_rrf([([], 1.0), (["a"], 1.0)], rrf_k=60) == ["a"]
```

- [ ] **Step 2: 运行测试确认失败**

```bash
conda run -n agent python -m pytest tests/retrieval/test_rrf.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'retrieval.rrf'`

- [ ] **Step 3: 实现 weighted_rrf**

```python
# src/retrieval/rrf.py
"""Reciprocal Rank Fusion utilities."""


def weighted_rrf(
    rankings: list[tuple[list[str], float]],
    rrf_k: int = 60,
) -> list[str]:
    """Fuse multiple weighted rankings into a single ordered citation list.

    score(c) += weight / (rrf_k + rank + 1)  for each appearance in a ranking.
    """
    scores: dict[str, float] = {}
    for ranked_list, weight in rankings:
        if not ranked_list or weight == 0:
            continue
        for rank, citation in enumerate(ranked_list):
            scores[citation] = scores.get(citation, 0.0) + weight / (rrf_k + rank + 1)

    return sorted(scores, key=lambda c: scores[c], reverse=True)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
conda run -n agent python -m pytest tests/retrieval/test_rrf.py -v
```

Expected: 5 passed

---

### Task 2: 双索引构建

**Files:**
- Modify: `src/indexing/build_bm25.py`

- [ ] **Step 1: 重构 build_bm25.py 支持按 source 构建**

将现有 `main()` 拆为 `_build_index(docs, output_dir, label)` 辅助函数，然后：

```python
def _build_index(docs: list[dict], output_dir: str, label: str) -> None:
  os.makedirs(output_dir, exist_ok=True)
  indexed_texts = [doc["indexed_text"] for doc in docs]
  citations     = [doc["citation"] for doc in docs]
  print(f"Tokenizing {label} ({len(docs):,} docs) ...")
  tokenized = tokenize_for_bm25(indexed_texts, citations=citations, show_progress=True)
  retriever = bm25s.BM25()
  retriever.index(tokenized, show_progress=True)
  retriever.save(output_dir, corpus=indexed_texts)
  print(f"{label} index saved → {output_dir}/")


def main() -> None:
  # load corpus.pkl
  court_docs = [d for d in corpus if d["source"] == "court"]
  law_docs   = [d for d in corpus if d["source"] == "law"]
  _build_index(court_docs, os.path.join(INDEX_DIR, "bm25_court"), "court")
  _build_index(law_docs,   os.path.join(INDEX_DIR, "bm25_law"),   "law")
```

删除对旧 `indexes/bm25/` 的写入。

- [ ] **Step 2: 在 sample 索引上验证构建**

若 `indexes/corpus.pkl` 不存在，先用 sample 数据或跳过全量构建，仅验证脚本语法：

```bash
conda run -n agent python -c "import ast; ast.parse(open('src/indexing/build_bm25.py').read())"
```

全量构建（用户自行执行，耗时 15-40 分钟）：

```bash
conda run -n agent python src/indexing/build_bm25.py
```

Expected: 产出 `indexes/bm25_court/` 和 `indexes/bm25_law/`

---

### Task 3: 双索引检索 + RRF 融合

**Files:**
- Modify: `src/retrieval/bm25.py`

- [ ] **Step 1: 扩展全局状态与加载逻辑**

```python
BM25_COURT_DIR = os.path.join(INDEX_DIR, "bm25_court")
BM25_LAW_DIR   = os.path.join(INDEX_DIR, "bm25_law")

_retriever_court: bm25s.BM25 | None = None
_retriever_law:   bm25s.BM25 | None = None
_corpus_court:    list[dict] | None  = None
_corpus_law:      list[dict] | None  = None
```

`_load_index()` 加载两个 retriever；同时从 `corpus.pkl` 按 source 分出 `_corpus_court` / `_corpus_law` 用于 index→citation 映射。若目录不存在则 `raise FileNotFoundError` 并提示运行 `build_bm25.py`。

- [ ] **Step 2: 重写 retrieve_bm25 签名与逻辑**

```python
from retrieval.rrf import weighted_rrf

def retrieve_bm25(
    query: str,
    k: int = 200,
    k_court: int = 300,
    k_law: int = 300,
    weight_extracted: float = 2.0,
    weight_law: float = 1.2,
    weight_court: float = 1.0,
    rrf_k: int = 60,
) -> list[str]:
    _load_index()

    extracted = extract_citations_from_query(query)
    tokenized_q = tokenize_for_bm25(
        [query], citations=[extracted], show_progress=False
    )

    court_results, _ = _retriever_court.retrieve(tokenized_q, k=k_court)
    law_results, _   = _retriever_law.retrieve(tokenized_q, k=k_law)

    court_citations = [_corpus_court[i]["citation"] for i in court_results[0].tolist()]
    law_citations   = [_corpus_law[i]["citation"] for i in law_results[0].tolist()]

    rankings: list[tuple[list[str], float]] = []
    if extracted:
        rankings.append((extracted, weight_extracted))
    if court_citations:
        rankings.append((court_citations, weight_court))
    if law_citations:
        rankings.append((law_citations, weight_law))

    fused = weighted_rrf(rankings, rrf_k=rrf_k)
    return fused[:k]
```

- [ ] **Step 3: 手动冒烟测试（需已构建双索引）**

```bash
conda run -n agent python -c "
from retrieval.bm25 import retrieve_bm25
r = retrieve_bm25('Art. 8 Abs. 1 ATSG invalidity', k=10)
print(len(r), r[:5])
"
```

Expected: 返回 ≤10 条 citation，含 law 类结果

---

### Task 4: Query CLI 参数扩展

**Files:**
- Modify: `src/query/run.py`

- [ ] **Step 1: 扩展 predict_citations 和 run()**

```python
def predict_citations(
    query: str,
    k: int = 200,
    k_court: int = 300,
    k_law: int = 300,
    weight_extracted: float = 2.0,
    weight_law: float = 1.2,
    weight_court: float = 1.0,
    rrf_k: int = 60,
) -> list[str]:
    return retrieve_bm25(
        query, k=k, k_court=k_court, k_law=k_law,
        weight_extracted=weight_extracted, weight_law=weight_law,
        weight_court=weight_court, rrf_k=rrf_k,
    )
```

- [ ] **Step 2: 添加 CLI 参数**

```python
parser.add_argument("--k-court", type=int, default=300)
parser.add_argument("--k-law", type=int, default=300)
parser.add_argument("--k", type=int, default=200)
parser.add_argument("--weight-extracted", type=float, default=2.0)
parser.add_argument("--weight-law", type=float, default=1.2)
parser.add_argument("--weight-court", type=float, default=1.0)
parser.add_argument("--rrf-k", type=int, default=60)
```

`run()` 和 `main()` 透传所有参数。

- [ ] **Step 3: 验证 CLI help**

```bash
conda run -n agent python src/query/run.py --help
```

Expected: 显示所有新参数及默认值

---

### Task 5: 端到端验证

- [ ] **Step 1: 运行全量测试**

```bash
conda run -n agent python -m pytest tests/ -v
```

Expected: 全部通过（含新 RRF 测试 + 原有 eval 测试）

- [ ] **Step 2: query → eval 端到端（需双索引已构建）**

```bash
conda run -n agent python src/query/run.py --output results/pred_split_rrf.csv
conda run -n agent python src/eval/macro_f1.py --predictions results/pred_split_rrf.csv
```

Expected: 输出 Macro F1 报告，无报错

- [ ] **Step 3: 更新 docs/plan/README.md 索引行**

添加 `bm25_split_rrf.plan.md` 条目，状态 pending → 已实现（完成后更新）

---

## 注意事项

- 双索引全量构建需用户在有 `indexes/corpus.pkl` 的环境执行，耗时较长
- 数值与旧单索引 baseline 不同属预期，不追求数值一致
- `macro_f1.py` 无需修改
