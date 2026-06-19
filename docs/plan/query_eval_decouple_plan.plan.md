---
name: query-eval-decouple
overview: 将 query 流程与 eval 解耦，predictions CSV 作为唯一接口
todos:
  - id: retrieval-module
    content: 新建 src/retrieval/bm25.py，从 macro_f1.py 迁出检索逻辑
    status: pending
  - id: query-cli
    content: 新建 src/query/run.py，读 input CSV 写 predictions CSV
    status: pending
  - id: eval-refactor
    content: 重构 src/eval/macro_f1.py 为纯 eval（无检索依赖）
    status: pending
  - id: gitignore
    content: .gitignore 增加 results/
    status: pending
  - id: verify
    content: 端到端验证 query → eval Macro F1 与重构前一致
    status: pending
---

# Query / Eval 解耦 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 BM25 query 流程与 Macro F1 eval 解耦，通过 `results/predictions.csv`（`query_id,predicted_citations`）作为唯一接口。

**Architecture:** 检索逻辑迁入 `src/retrieval/bm25.py`；`src/query/run.py` 读 query CSV、调 `predict_citations()`、写 predictions；`src/eval/macro_f1.py` 只读 predictions + `dataset/val.csv` gold 算分。Spec：[docs/superpowers/specs/2026-06-19-query-eval-decouple-design.md](../superpowers/specs/2026-06-19-query-eval-decouple-design.md)

**Tech Stack:** Python 3, bm25s, csv, conda env `agent`

---

## File Map

| 文件 | 操作 | 职责 |
|------|------|------|
| `src/retrieval/bm25.py` | 新建 | BM25 索引加载、citation regex、retrieve_bm25 |
| `src/query/run.py` | 新建 | Query CLI，写 predictions CSV |
| `src/eval/macro_f1.py` | 重构 | 纯指标 + eval CLI |
| `tests/eval/test_macro_f1.py` | 新建 | parse_citations、F1 单元测试 |
| `.gitignore` | 修改 | 加 `results/` |
| `docs/plan/README.md` | 修改 | 索引本 plan |

---

### Task 1: Eval 纯函数 + 单元测试

**Files:**
- Create: `tests/eval/test_macro_f1.py`
- Modify: `src/eval/macro_f1.py`（先只加函数，检索代码暂留）

- [ ] **Step 1: 创建测试目录与测试文件**

```bash
mkdir -p tests/eval
```

Create `tests/eval/test_macro_f1.py`:

```python
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from eval.macro_f1 import (
    _query_f1,
    compute_macro_f1,
    parse_citations,
)


def test_parse_citations_splits_and_strips():
    s = "Art. 23 OR; Art. 24 OR ;BGE 136 III 528"
    assert parse_citations(s) == {
        "Art. 23 OR",
        "Art. 24 OR",
        "BGE 136 III 528",
    }


def test_parse_citations_empty():
    assert parse_citations("") == set()
    assert parse_citations("  ;  ") == set()


def test_query_f1_exact_match():
    gold = {"Art. 23 OR", "Art. 24 OR", "BGE 136 III 528 E. 3.4.1"}
    pred = {"Art. 23 OR", "Art. 24 OR", "Art. 31 OR"}
    p, r, f1 = _query_f1(gold, pred)
    assert p == 2 / 3
    assert r == 2 / 3
    assert abs(f1 - 2 / 3) < 1e-9


def test_query_f1_empty_pred():
    p, r, f1 = _query_f1({"Art. 1 OR"}, set())
    assert (p, r, f1) == (0.0, 0.0, 0.0)


def test_compute_macro_f1():
    gold_list = [{"a"}, {"x", "y"}]
    pred_list = [{"a"}, {"x"}]
    assert compute_macro_f1(gold_list, pred_list) == (1.0 + 2 / 3) / 2
```

- [ ] **Step 2: 运行测试，确认 parse_citations 失败**

Run:
```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python -m pytest tests/eval/test_macro_f1.py -v
```
Expected: FAIL — `ImportError` 或 `cannot import name 'parse_citations'`

- [ ] **Step 3: 在 macro_f1.py 添加 parse_citations（检索代码不动）**

在 `src/eval/macro_f1.py` 的 Metrics 区块加入：

```python
def parse_citations(s: str) -> set[str]:
    """Parse semicolon-separated citation string into a set."""
    return {c.strip() for c in s.split(";") if c.strip()}
```

- [ ] **Step 4: 再次运行测试**

Run:
```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python -m pytest tests/eval/test_macro_f1.py -v
```
Expected: 5 passed

---

### Task 2: 新建 retrieval 模块

**Files:**
- Create: `src/retrieval/bm25.py`

- [ ] **Step 1: 创建 `src/retrieval/bm25.py`**

从 `src/eval/macro_f1.py` 剪切以下内容到新文件（保持逻辑不变）：

- 路径常量：`ROOT_DIR`, `INDEX_DIR`, `BM25_DIR`, `CORPUS_PATH`
- Citation regex：`STATUTE_RE`, `BGE_RE`, `BGER_RE`
- 函数：`extract_citations_from_query`, `_load_index`, `retrieve_bm25`
- 全局：`_retriever`, `_corpus`
- imports：`os`, `pickle`, `re`, `sys`, `bm25s`, `tokenize_for_bm25`

新文件完整内容：

```python
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
```

- [ ] **Step 2: 验证 retrieval 模块可导入**

Run:
```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python -c "
import sys; sys.path.insert(0, 'src')
from retrieval.bm25 import retrieve_bm25
print('import ok')
"
```
Expected: `import ok`（若 indexes 不存在，import 仍应成功；retrieve 运行时才需索引）

---

### Task 3: 新建 Query CLI

**Files:**
- Create: `src/query/run.py`

- [ ] **Step 1: 创建 `src/query/run.py`**

```python
"""
Run citation prediction pipeline on a query CSV and write predictions.

Usage:
    conda run -n agent python src/query/run.py
    conda run -n agent python src/query/run.py --input dataset/test.csv --output results/test_predictions.csv
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from retrieval.bm25 import retrieve_bm25

ROOT_DIR    = os.path.join(os.path.dirname(__file__), "../..")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
DEFAULT_INPUT  = os.path.join(DATASET_DIR, "val.csv")
DEFAULT_OUTPUT = os.path.join(ROOT_DIR, "results/predictions.csv")


def predict_citations(query: str, k: int = 200) -> list[str]:
    """Baseline pipeline: BM25 top-k as final prediction."""
    return retrieve_bm25(query, k=k)


def format_citations(citations: list[str]) -> str:
    return ";".join(citations)


def load_queries(path: str) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    fieldnames = rows[0].keys()
    for col in ("query_id", "query"):
        if col not in fieldnames:
            raise ValueError(f"Input CSV missing required column: {col}")
    return rows


def write_predictions(path: str, rows: list[tuple[str, str]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "predicted_citations"])
        writer.writeheader()
        for qid, pred in rows:
            writer.writerow({"query_id": qid, "predicted_citations": pred})


def run(input_path: str, output_path: str, k: int) -> None:
    queries = load_queries(input_path)
    results: list[tuple[str, str]] = []
    for i, row in enumerate(queries, 1):
        qid = row["query_id"]
        print(f"[{i}/{len(queries)}] {qid}", file=sys.stderr)
        citations = predict_citations(row["query"], k=k)
        results.append((qid, format_citations(citations)))
    write_predictions(output_path, results)
    print(f"Wrote {len(results)} predictions → {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run query pipeline and write predictions CSV")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input query CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output predictions CSV")
    parser.add_argument("--k", type=int, default=200, help="BM25 top-k for baseline pipeline")
    args = parser.parse_args()
    run(args.input, args.output, k=args.k)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 在 sample 索引上试跑（若有 indexes_sample）**

若本地只有 `indexes_sample/`，先确认 `indexes/` 存在；否则跳过此步，在 Task 5 用完整索引验证。

Run:
```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python src/query/run.py \
  --input dataset/val.csv \
  --output results/predictions.csv \
  --k 200
```
Expected: stderr 显示 10 条 query 进度，生成 `results/predictions.csv`（10 行 + header）

- [ ] **Step 3: 检查 predictions 格式**

Run:
```bash
head -n 3 results/predictions.csv
```
Expected: `query_id,predicted_citations` 及 `val_001,...` 行，citations 以 `;` 分隔

---

### Task 4: 重构 Eval 模块

**Files:**
- Modify: `src/eval/macro_f1.py`（删除检索，新增 load/evaluate）

- [ ] **Step 1: 删除 macro_f1.py 中的检索相关代码**

删除：
- `import pickle`, `import re`, `import bm25s`
- `INDEX_DIR`, `BM25_DIR`, `CORPUS_PATH`
- `STATUTE_RE`, `BGE_RE`, `BGER_RE`
- `extract_citations_from_query`
- `compute_recall_at_k`
- `_retriever`, `_corpus`, `_load_index`, `retrieve_bm25`
- `evaluate_val` 旧实现（含检索循环）

- [ ] **Step 2: 替换为纯 eval 实现**

`src/eval/macro_f1.py` 重构后核心内容：

```python
"""
Evaluate predictions CSV against val.csv gold using Macro F1 (exact string match).

Usage:
    conda run -n agent python src/eval/macro_f1.py
    conda run -n agent python src/eval/macro_f1.py --predictions results/predictions.csv
    conda run -n agent python src/eval/macro_f1.py --output results/eval_report.txt
"""

import argparse
import csv
import os
import sys

ROOT_DIR    = os.path.join(os.path.dirname(__file__), "../..")
DATASET_DIR = os.path.join(ROOT_DIR, "dataset")
VAL_CSV = os.path.join(DATASET_DIR, "val.csv")
DEFAULT_PREDICTIONS = os.path.join(ROOT_DIR, "results/predictions.csv")


def parse_citations(s: str) -> set[str]:
    return {c.strip() for c in s.split(";") if c.strip()}


def _query_f1(gold: set[str], pred: set[str]) -> tuple[float, float, float]:
    if not pred:
        return 0.0, 0.0, 0.0
    tp = len(gold & pred)
    p  = tp / len(pred)
    r  = tp / len(gold) if gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def compute_macro_f1(gold_list: list[set[str]], pred_list: list[set[str]]) -> float:
    scores = [_query_f1(g, p)[2] for g, p in zip(gold_list, pred_list)]
    return sum(scores) / len(scores) if scores else 0.0


def load_gold(path: str = VAL_CSV) -> list[tuple[str, set[str]]]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        (row["query_id"], parse_citations(row["gold_citations"]))
        for row in rows
    ]


def load_predictions(path: str) -> dict[str, set[str]]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for col in ("query_id", "predicted_citations"):
        if rows and col not in rows[0]:
            raise ValueError(f"Predictions CSV missing required column: {col}")
    pred_map = {row["query_id"]: parse_citations(row["predicted_citations"]) for row in rows}
    return pred_map


def evaluate_predictions(predictions_path: str, gold_path: str = VAL_CSV) -> dict:
    gold_rows = load_gold(gold_path)
    pred_map = load_predictions(predictions_path)

    extra_ids = set(pred_map) - {qid for qid, _ in gold_rows}
    if extra_ids:
        print(f"Warning: ignoring {len(extra_ids)} query_id(s) not in gold: {sorted(extra_ids)}", file=sys.stderr)

    missing = [qid for qid, _ in gold_rows if qid not in pred_map]
    if missing:
        raise ValueError(f"Predictions missing {len(missing)} query_id(s) from gold: {missing}")

    results = []
    for qid, gold in gold_rows:
        pred = pred_map[qid]
        p, r, f1 = _query_f1(gold, pred)
        tp = len(gold & pred)
        missed = sorted(gold - pred)
        results.append({
            "query_id": qid,
            "gold": len(gold),
            "pred": len(pred),
            "tp": tp,
            "precision": p,
            "recall": r,
            "f1": f1,
            "missed": missed,
        })

    return {"rows": results}


def print_results(data: dict, output_path: str | None = None) -> None:
    rows = data["rows"]
    header = (
        f"{'query_id':<12} | {'gold':>4} | {'pred':>4} | {'tp':>4} | "
        f"{'precision':>9} | {'recall':>6} | {'f1':>8} | missed (first 3)"
    )
    sep = "-" * len(header)
    lines = [header, sep]

    for r in rows:
        missed_preview = str(r["missed"][:3])[1:-1] if r["missed"] else "-"
        lines.append(
            f"{r['query_id']:<12} | {r['gold']:>4} | {r['pred']:>4} | {r['tp']:>4} | "
            f"{r['precision']:>9.4f} | {r['recall']:>6.4f} | {r['f1']:>8.4f} | {missed_preview}"
        )

    lines.append(sep)
    macro_f1 = sum(r["f1"] for r in rows) / len(rows) if rows else 0.0
    lines.append(f"{'AGGREGATE':<12} | {'':>4} | {'':>4} | {'':>4} | {'':>9} | {'':>6} | {macro_f1:>8.4f} |")

    output = "\n".join(lines)
    print(output)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output + "\n")
        print(f"\nResults saved → {output_path}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions CSV against val.csv gold")
    parser.add_argument("--predictions", default=DEFAULT_PREDICTIONS, help="Predictions CSV path")
    parser.add_argument("--output", default=None, help="Optional path to save text report")
    args = parser.parse_args()

    print(f"Evaluating {args.predictions} against {VAL_CSV}\n")
    data = evaluate_predictions(args.predictions)
    print_results(data, output_path=args.output)


if __name__ == "__main__":
    main()
```

注意：Step 2 代码块中 `load_predictions` 的 `for col` 行缩进应为 4 空格（粘贴时校对）。

- [ ] **Step 3: 运行单元测试**

Run:
```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python -m pytest tests/eval/test_macro_f1.py -v
```
Expected: 5 passed

- [ ] **Step 4: 用已有 predictions 跑 eval**

Run:
```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python src/eval/macro_f1.py --predictions results/predictions.csv
```
Expected: 打印 10 条明细 + AGGREGATE Macro F1

---

### Task 5: .gitignore 与文档

**Files:**
- Modify: `.gitignore`
- Modify: `docs/plan/README.md`
- Modify: `docs/superpowers/specs/2026-06-19-query-eval-decouple-design.md`（状态改为已实现）

- [ ] **Step 1: 更新 .gitignore**

在 `.gitignore` 末尾添加：
```
results/
```

- [ ] **Step 2: 更新 docs/plan/README.md 索引行**

```markdown
| [query_eval_decouple_plan.plan.md](query_eval_decouple_plan.plan.md) | Query / Eval 解耦：predictions CSV 接口 | 待实现 |
```

- [ ] **Step 3: 更新 spec 状态**

将 `docs/superpowers/specs/2026-06-19-query-eval-decouple-design.md` 中 `**状态:** 待审阅` 改为 `**状态:** 已实现`

---

### Task 6: 端到端验证

- [ ] **Step 1: 记录重构前 baseline Macro F1（若尚未记录）**

在完整 `indexes/` 存在的前提下，用旧代码或等价逻辑记录 AGGREGATE F1@200。若仓库已重构，用当前 query+eval 跑一次作为 baseline。

Run:
```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python src/query/run.py --k 200 --output results/predictions.csv
/Users/vingo/opt/anaconda3/envs/agent/bin/python src/eval/macro_f1.py --predictions results/predictions.csv --output results/eval_report.txt
```

- [ ] **Step 2: 验证缺失 query_id 报错**

创建只有 1 行的残缺 predictions，确认 eval 以 `ValueError` 退出并列出缺失 id。

- [ ] **Step 3: 验证 test.csv 可跑 query（无 eval）**

Run:
```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python src/query/run.py \
  --input dataset/test.csv \
  --output results/test_predictions.csv
```
Expected: 生成 test predictions，行数与 test.csv 一致

---

## Spec Coverage Checklist

| Spec 要求 | Task |
|-----------|------|
| `src/retrieval/bm25.py` | Task 2 |
| `src/query/run.py` + `--input/--output/--k` | Task 3 |
| predictions CSV 格式 | Task 3 |
| `src/eval/macro_f1.py` 纯 eval + `--predictions` | Task 4 |
| gold 固定 val.csv | Task 4 `load_gold` |
| 缺失 query_id 报错 / 多余警告 | Task 4 `evaluate_predictions` |
| 去掉 Recall@k | Task 4（删除 `compute_recall_at_k`） |
| `results/` gitignore | Task 5 |
| 端到端验证 k=200 | Task 6 |
