---
name: query-rewrite
overview: 在 query pipeline 第一步加入 LLM query rewrite，输出 spec §7.1 结构化结果，当前用 format_search_text(de) 增强 BM25 召回
todos:
  - id: deps
    content: 安装 openai、python-dotenv 到 conda agent 环境
    status: completed
  - id: llm-client
    content: 新建 src/llm/client.py + 单元测试
    status: completed
  - id: rewrite-module
    content: 新建 src/query/rewrite.py（parse/format/rewrite_query）+ 单元测试
    status: completed
  - id: bm25-search-text
    content: retrieve_bm25 增加 search_text 参数
    status: completed
  - id: query-cli
    content: run.py 集成 rewrite、--no-rewrite、--rewrite-log
    status: completed
  - id: verify
    content: pytest + 端到端 query → eval 对比
    status: completed
isProject: false
---

# LLM Query Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 query pipeline 第一步调用 LLM 将英文法律问题 rewrite 为结构化 `RewriteResult`，经 `format_search_text(lang="de")` 生成检索文本喂 BM25；citation 提取仍用原文。

**Architecture:** `src/llm/client.py` 封装 OpenAI 兼容 `chat_json()`；`src/query/rewrite.py` 负责 prompt、schema 校验、`format_search_text()`；`retrieve_bm25()` 新增 `search_text` 与原文 `query` 分离；`run.py` 默认开启 rewrite，`--no-rewrite` 回退 baseline。

**Tech Stack:** Python 3, openai, python-dotenv, pytest, conda env `agent`

**设计文档:** [docs/superpowers/specs/2026-06-19-query-rewrite-design.md](../superpowers/specs/2026-06-19-query-rewrite-design.md)

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `src/llm/__init__.py` | 新建 | 包标记 |
| `src/llm/client.py` | 新建 | `.env` 加载、`chat_json()` |
| `src/query/rewrite.py` | 新建 | `RewriteResult`、`parse_rewrite_result`、`format_search_text`、`rewrite_query` |
| `src/retrieval/bm25.py` | 修改 | `search_text` 参数 |
| `src/query/run.py` | 修改 | rewrite 集成、CLI |
| `tests/query/test_rewrite.py` | 新建 | rewrite 单元测试 |
| `tests/llm/test_client.py` | 新建 | client 单元测试（mock API） |
| `docs/plan/README.md` | 修改 | 索引本 plan |

---

### Task 1: 安装依赖

**Files:**
- Modify: 项目根目录（pip 安装，无新文件）

- [ ] **Step 1: 安装 openai 和 python-dotenv**

```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/pip install openai python-dotenv
```

Expected: `Successfully installed openai-... python-dotenv-...`

- [ ] **Step 2: 验证 import**

```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python -c "import openai; import dotenv; print('ok')"
```

Expected: `ok`

---

### Task 2: LLM Client

**Files:**
- Create: `src/llm/__init__.py`
- Create: `src/llm/client.py`
- Create: `tests/llm/test_client.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/llm/test_client.py
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from llm import client as llm_client


def test_chat_json_parses_response():
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"key": "value"}'))
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    with patch.object(llm_client, "_get_client", return_value=mock_client):
        result = llm_client.chat_json("system", "user")

    assert result == {"key": "value"}
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["response_format"] == {"type": "json_object"}


def test_get_client_raises_when_env_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL_ID", raising=False)
    llm_client._client = None  # reset cached client

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        llm_client._get_client()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
cd /Users/vingo/Desktop/github_repos/LawAgent
/Users/vingo/opt/anaconda3/envs/agent/bin/python -m pytest tests/llm/test_client.py -v
```

Expected: FAIL（`llm.client` 不存在）

- [ ] **Step 3: 实现 client**

```python
# src/llm/__init__.py
# (empty file)
```

```python
# src/llm/client.py
"""OpenAI-compatible LLM client."""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

ROOT_DIR = os.path.join(os.path.dirname(__file__), "../..")
load_dotenv(os.path.join(ROOT_DIR, ".env"))

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_id = os.getenv("OPENAI_MODEL_ID")

    missing = [
        name
        for name, val in [
            ("OPENAI_API_KEY", api_key),
            ("OPENAI_BASE_URL", base_url),
            ("OPENAI_MODEL_ID", model_id),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

    _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def chat_json(system: str, user: str) -> dict:
    """Call OpenAI-compatible API with JSON response format."""
    client = _get_client()
    model_id = os.environ["OPENAI_MODEL_ID"]
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM returned empty content")
    return json.loads(content)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python -m pytest tests/llm/test_client.py -v
```

Expected: PASS

---

### Task 3: Rewrite 纯函数（parse + format）

**Files:**
- Create: `src/query/rewrite.py`（先写 dataclass + parse + format，不含 rewrite_query）
- Create: `tests/query/test_rewrite.py`

- [ ] **Step 1: 写失败测试**

```python
# tests/query/test_rewrite.py
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from query.rewrite import format_search_text, parse_rewrite_result, RewriteResult


def _sample_data():
    return {
        "legal_issue": "Untersuchungshaft Verlängerung",
        "expected_codes": ["StPO", "StGB"],
        "search_terms": {
            "de": ["Kollusionsgefahr", "Verhältnismäßigkeit"],
            "fr": ["prolongation détention"],
        },
    }


def test_parse_rewrite_result_valid():
    result = parse_rewrite_result(_sample_data())
    assert isinstance(result, RewriteResult)
    assert result.legal_issue == "Untersuchungshaft Verlängerung"
    assert result.expected_codes == ["StPO", "StGB"]
    assert result.search_terms["de"] == ["Kollusionsgefahr", "Verhältnismäßigkeit"]
    assert result.search_terms["fr"] == ["prolongation détention"]


def test_parse_rewrite_result_missing_fr_defaults_empty():
    data = _sample_data()
    del data["search_terms"]["fr"]
    result = parse_rewrite_result(data)
    assert result.search_terms["fr"] == []


def test_parse_rewrite_result_missing_de_raises():
    data = _sample_data()
    del data["search_terms"]["de"]
    with pytest.raises(ValueError, match="search_terms.de"):
        parse_rewrite_result(data)


def test_parse_rewrite_result_empty_de_raises():
    data = _sample_data()
    data["search_terms"]["de"] = []
    with pytest.raises(ValueError, match="search_terms.de"):
        parse_rewrite_result(data)


def test_format_search_text_default_de():
    result = parse_rewrite_result(_sample_data())
    text = format_search_text(result)
    assert text == (
        "Untersuchungshaft Verlängerung "
        "Kollusionsgefahr Verhältnismäßigkeit"
    )


def test_format_search_text_lang_fr():
    result = parse_rewrite_result(_sample_data())
    text = format_search_text(result, lang="fr")
    assert text == "Untersuchungshaft Verlängerung prolongation détention"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python -m pytest tests/query/test_rewrite.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现 parse + format**

```python
# src/query/rewrite.py
"""LLM-based query rewriting for legal retrieval."""

from dataclasses import dataclass


@dataclass
class RewriteResult:
    legal_issue: str
    expected_codes: list[str]
    search_terms: dict[str, list[str]]


def parse_rewrite_result(data: dict) -> RewriteResult:
    legal_issue = data.get("legal_issue", "")
    if not isinstance(legal_issue, str) or not legal_issue.strip():
        raise ValueError("legal_issue must be a non-empty string")

    expected_codes = data.get("expected_codes", [])
    if not isinstance(expected_codes, list):
        raise ValueError("expected_codes must be a list")
    expected_codes = [str(c) for c in expected_codes]

    search_terms = data.get("search_terms")
    if not isinstance(search_terms, dict):
        raise ValueError("search_terms must be a dict")
    if "de" not in search_terms:
        raise ValueError("search_terms.de is required")
    de_terms = search_terms["de"]
    if not isinstance(de_terms, list) or not de_terms:
        raise ValueError("search_terms.de must be a non-empty list")
    de_terms = [str(t) for t in de_terms]

    fr_terms = search_terms.get("fr", [])
    if fr_terms is None:
        fr_terms = []
    if not isinstance(fr_terms, list):
        raise ValueError("search_terms.fr must be a list")
    fr_terms = [str(t) for t in fr_terms]

    return RewriteResult(
        legal_issue=legal_issue.strip(),
        expected_codes=expected_codes,
        search_terms={"de": de_terms, "fr": fr_terms},
    )


def format_search_text(result: RewriteResult, lang: str = "de") -> str:
    """Join legal_issue + search_terms[lang] into a single search string."""
    parts = list(result.search_terms.get(lang, []))
    if result.legal_issue:
        parts = [result.legal_issue] + parts
    return " ".join(parts)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python -m pytest tests/query/test_rewrite.py -v
```

Expected: PASS（6 tests）

---

### Task 4: rewrite_query（LLM 调用 + prompt）

**Files:**
- Modify: `src/query/rewrite.py`
- Modify: `tests/query/test_rewrite.py`

- [ ] **Step 1: 追加失败测试**

在 `tests/query/test_rewrite.py` 末尾追加：

```python
from unittest.mock import patch

from query.rewrite import rewrite_query


def test_rewrite_query_calls_llm_and_parses():
    mock_data = {
        "legal_issue": "Test Issue",
        "expected_codes": ["OR"],
        "search_terms": {"de": ["Term A"], "fr": []},
    }
    with patch("query.rewrite.chat_json", return_value=mock_data) as mock_chat:
        result = rewrite_query("Can a party rescind a contract?")

    assert result.legal_issue == "Test Issue"
    mock_chat.assert_called_once()
    system_prompt, user_prompt = mock_chat.call_args[0]
    assert "JSON" in system_prompt
    assert "Can a party rescind a contract?" in user_prompt
```

- [ ] **Step 2: 运行新测试确认失败**

```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python -m pytest tests/query/test_rewrite.py::test_rewrite_query_calls_llm_and_parses -v
```

Expected: FAIL（`rewrite_query` 未定义）

- [ ] **Step 3: 实现 rewrite_query**

在 `src/query/rewrite.py` 追加：

```python
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm.client import chat_json  # noqa: E402

import os

SYSTEM_PROMPT = """You are a Swiss legal retrieval assistant.

Analyze the English legal question and output a single JSON object with these fields:
- "legal_issue": core legal issue in German (non-empty string)
- "expected_codes": list of Swiss code abbreviations that may apply (e.g. StPO, StGB, OR, ZGB, ATSG, IVG); may be empty
- "search_terms": object with:
  - "de": 5-10 German legal search phrases (Swiss legal terminology, natural casing, no stemming/stopword removal)
  - "fr": 3-5 French legal search phrases (may be empty list)

Output natural German/French legal terms. Do NOT lowercase, stem, or remove stopwords.
Output valid JSON only, matching the schema exactly."""


def rewrite_query(query: str) -> RewriteResult:
    data = chat_json(SYSTEM_PROMPT, query)
    if not isinstance(data, dict):
        raise ValueError(f"LLM response must be a JSON object, got: {type(data)}")
    try:
        return parse_rewrite_result(data)
    except ValueError as exc:
        raise ValueError(
            f"Invalid rewrite schema: {exc}\nRaw JSON: {json.dumps(data, ensure_ascii=False)}"
        ) from exc
```

注意：将 `import os` 和 `sys.path` 移到文件顶部，与现有 imports 合并（实现时整理 import 顺序）。

- [ ] **Step 4: 运行全部 rewrite 测试**

```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python -m pytest tests/query/test_rewrite.py -v
```

Expected: PASS（7 tests）

---

### Task 5: retrieve_bm25 增加 search_text

**Files:**
- Modify: `src/retrieval/bm25.py:68-84`

- [ ] **Step 1: 修改 retrieve_bm25 签名与 tokenize 输入**

将 `retrieve_bm25` 函数签名和 tokenize 部分改为：

```python
def retrieve_bm25(
    query: str,
    search_text: str | None = None,
    k: int = 200,
    k_court: int = 300,
    k_law: int = 300,
    weight_extracted: float = 2.0,
    weight_law: float = 1.2,
    weight_court: float = 1.0,
    rrf_k: int = 60,
) -> list[str]:
    """Return up to k citation strings via dual BM25 + query extraction RRF fusion."""
    _load_index()

    extracted = extract_citations_from_query(query)
    text_for_search = search_text if search_text is not None else query
    tokenized_q = tokenize_for_bm25(
        [text_for_search], citations=[extracted], show_progress=False
    )
    # ... rest unchanged
```

- [ ] **Step 2: 确认现有测试仍通过**

```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python -m pytest tests/ -v
```

Expected: 全部 PASS（`search_text` 默认 None，行为与改前一致）

---

### Task 6: run.py 集成 rewrite

**Files:**
- Modify: `src/query/run.py`

- [ ] **Step 1: 更新 predict_citations**

```python
from query.rewrite import format_search_text, rewrite_query

def predict_citations(
    query: str,
    use_rewrite: bool = True,
    k: int = 200,
    k_court: int = 300,
    k_law: int = 300,
    weight_extracted: float = 2.0,
    weight_law: float = 1.2,
    weight_court: float = 1.0,
    rrf_k: int = 60,
) -> list[str]:
  search_text = None
  if use_rewrite:
      result = rewrite_query(query)
      search_text = format_search_text(result, lang="de")
  return retrieve_bm25(
      query,
      search_text=search_text,
      k=k,
      k_court=k_court,
      k_law=k_law,
      weight_extracted=weight_extracted,
      weight_law=weight_law,
      weight_court=weight_court,
      rrf_k=rrf_k,
  )
```

- [ ] **Step 2: 更新 run() 支持 rewrite_log**

```python
import json

def _log_rewrite(log_dir: str, query_id: str, result) -> None:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{query_id}.json")
    payload = {
        "query_id": query_id,
        "legal_issue": result.legal_issue,
        "expected_codes": result.expected_codes,
        "search_terms": result.search_terms,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run(
    input_path: str,
    output_path: str,
    k: int,
    k_court: int,
    k_law: int,
    weight_extracted: float,
    weight_law: float,
    weight_court: float,
    rrf_k: int,
    use_rewrite: bool = True,
    rewrite_log_dir: str | None = None,
) -> None:
    queries = load_queries(input_path)
    results: list[tuple[str, str]] = []
    for i, row in enumerate(queries, 1):
        qid = row["query_id"]
        query = row["query"]
        print(f"[{i}/{len(queries)}] {qid}", file=sys.stderr)

        search_text = None
        if use_rewrite:
            rewrite_result = rewrite_query(query)
            search_text = format_search_text(rewrite_result, lang="de")
            if rewrite_log_dir:
                _log_rewrite(rewrite_log_dir, qid, rewrite_result)

        citations = retrieve_bm25(
            query,
            search_text=search_text,
            k=k,
            k_court=k_court,
            k_law=k_law,
            weight_extracted=weight_extracted,
            weight_law=weight_law,
            weight_court=weight_court,
            rrf_k=rrf_k,
        )
        results.append((qid, format_citations(citations)))
    write_predictions(output_path, results)
    print(f"Wrote {len(results)} predictions → {output_path}", file=sys.stderr)
```

说明：`run()` 内直接调 `rewrite_query` 以便写 log；`predict_citations` 仍保留供单测/复用。若希望 DRY，可让 `predict_citations` 接受可选 `RewriteResult`，但 YAGNI——当前 `run()` 展开即可。

- [ ] **Step 3: 更新 CLI**

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Run query pipeline and write predictions CSV")
    # ... existing args ...
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="Skip LLM query rewrite; use original query for BM25",
    )
    parser.add_argument(
        "--rewrite-log",
        default=None,
        help="Optional directory to write per-query rewrite JSON (debug only)",
    )
    args = parser.parse_args()
    run(
        args.input,
        args.output,
        k=args.k,
        k_court=args.k_court,
        k_law=args.k_law,
        weight_extracted=args.weight_extracted,
        weight_law=args.weight_law,
        weight_court=args.weight_court,
        rrf_k=args.rrf_k,
        use_rewrite=not args.no_rewrite,
        rewrite_log_dir=args.rewrite_log,
    )
```

- [ ] **Step 4: --no-rewrite 冒烟（不需 LLM）**

```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python src/query/run.py --no-rewrite --output results/pred_no_rewrite.csv
```

Expected: 正常写出 predictions（与改前 baseline 行为一致）

---

### Task 7: 端到端验证

**Files:**
- Modify: `docs/plan/README.md`（加索引行）
- Modify: `docs/superpowers/specs/2026-06-19-query-rewrite-design.md`（状态 → 已实现）

- [ ] **Step 1: 全量单元测试**

```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python -m pytest tests/ -v
```

Expected: 全部 PASS

- [ ] **Step 2: baseline eval**

```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python src/query/run.py --no-rewrite --output results/pred_no_rewrite.csv
/Users/vingo/opt/anaconda3/envs/agent/bin/python src/eval/macro_f1.py --predictions results/pred_no_rewrite.csv
```

记录 Macro F1 数值。

- [ ] **Step 3: rewrite eval（需 .env 与网络）**

```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python src/query/run.py \
  --output results/pred_rewrite.csv \
  --rewrite-log results/rewrite_logs/
/Users/vingo/opt/anaconda3/envs/agent/bin/python src/eval/macro_f1.py --predictions results/pred_rewrite.csv
```

Expected: 跑通 10 条 val；对比 Macro F1；抽查 `results/rewrite_logs/val_001.json` 德文词质量。

- [ ] **Step 4: 更新文档索引**

在 `docs/plan/README.md` 表格追加：

```markdown
| [query_rewrite.plan.md](query_rewrite.plan.md) | LLM Query Rewrite：pipeline 第一步 + BM25 search_text | 待实现 |
```

将 spec 状态改为 `已实现`。

---

## Spec 覆盖自检

| Spec 要求 | 对应 Task |
|-----------|-----------|
| `src/llm/client.py` + `chat_json()` | Task 2 |
| `RewriteResult` + schema 校验 | Task 3 |
| `format_search_text(lang)` | Task 3 |
| `rewrite_query()` + prompt | Task 4 |
| `retrieve_bm25(search_text=)` | Task 5 |
| 默认开启 rewrite | Task 6 |
| `--no-rewrite` | Task 6 |
| `--rewrite-log` | Task 6 |
| 不缓存 | 无缓存逻辑（符合） |
| 失败报错不 fallback | Task 4 `rewrite_query` 抛错；Task 6 不 catch |
| 单元测试 mock API | Task 2–4 |
| 依赖 openai + python-dotenv | Task 1 |

无遗漏。
