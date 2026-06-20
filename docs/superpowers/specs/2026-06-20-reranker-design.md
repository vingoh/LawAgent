# Reranker 集成设计

**日期：** 2026-06-20  
**状态：** 待实现  
**模型：** `models/bge-reranker-v2-m3`

---

## 1. 背景

当前 query pipeline 实现了 5 路加权 RRF hybrid retrieval（BM25 court/law + dense court/law + 直接抽取），最终输出 top-k citations。`bge-reranker-v2-m3` 模型已下载到本地，但尚未接入 pipeline。

当前 pipeline 结构：

```
hybrid RRF → rrf_result[:k] → 输出
```

目标：在 RRF 之后、最终截断之前，插入 cross-encoder reranker，对 top-100 候选精排，提升 Macro F1。

---

## 2. 目标

- 在现有 pipeline 的 RRF 输出之后插入 reranker 精排步骤
- Reranker 输入：top-100 RRF 候选
- Reranker query：原始英文 query + 德文 search_text 拼接（如果 rewrite 开启）
- Reranker 输出：重新排序后的 citation list，最终截断仍由现有 `--k` 参数控制
- 默认开启，可用 `--no-rerank` 关闭（类比现有 `--no-rewrite`）
- 新建独立 corpus 加载模块，避免多模块重复加载 corpus.pkl

---

## 3. 架构变化

### 3.1 Pipeline 流程（修改后）

```
Input English Query (CSV)
       │
       ├─► [optional] rewrite_query() → search_text (DE) + llm_articles
       │
       ├─► extract_citations_from_query(raw query)
       ├─► BM25 court  (search_text, k_court=300)
       ├─► BM25 law    (search_text, k_law=300)
       ├─► Dense court (search_text, if index exists)
       └─► Dense law   (search_text, if index exists)
                   │
             weighted_rrf(rankings, rrf_k=60)
                   │
             rrf_result[:rerank_top_k=100]
                   │
       [optional, default ON] rerank()
         query = raw_query + " " + search_text
         candidates = top-100 RRF 候选
         corpus_texts = citation→text 映射
                   │
             reranked_result[:k]
                   │
            "; "-joined predictions CSV
```

### 3.2 新增文件

| 文件 | 说明 |
|------|------|
| `src/retrieval/corpus.py` | 独立 corpus 加载模块，提供 `citation→text` 共享映射 |
| `src/retrieval/rerank.py` | Reranker 封装模块，懒加载 + 批量推理 |

### 3.3 修改文件

| 文件 | 改动内容 |
|------|---------|
| `src/retrieval/bm25.py` | corpus 加载逻辑移至 `corpus.py`，从 corpus 模块取 `_corpus_court` / `_corpus_law` |
| `src/retrieval/dense.py` | corpus 加载逻辑移至 `corpus.py`，从 corpus 模块取 `_corpus_court` / `_corpus_law` |
| `src/query/run.py` | 新增 reranker 调用逻辑、`--no-rerank` / `--rerank-top-k` / `--rerank-batch-size` CLI 参数、加载 corpus 模块 |

---

## 4. 模块设计

### 4.1 `src/retrieval/corpus.py`（新建）

职责：corpus.pkl 和 citation_to_idx.pkl 的唯一加载点，对外暴露 citation→text 映射。

```python
"""Shared corpus loader — loads corpus.pkl once and exposes citation→text mapping."""

_corpus_court: list[dict] | None = None  # source == "court" 的文档列表
_corpus_law:   list[dict] | None = None  # source == "law" 的文档列表
_corpus_texts: dict[str, str]   = {}     # citation -> indexed_text（按需构建）

def load_corpus() -> None:
    """加载 corpus.pkl 并按 source 字段分类，填充模块级变量（幂等）。"""

def get_corpus_court() -> list[dict]:
    """返回 court 文档列表（供 bm25.py 和 dense.py 使用）。"""

def get_corpus_law() -> list[dict]:
    """返回 law 文档列表（供 bm25.py 和 dense.py 使用）。"""

def get_corpus_texts() -> dict[str, str]:
    """返回 citation→indexed_text 映射（首次调用时从 corpus 列表构建）。"""
```

**加载路径：**
- `indexes/corpus.pkl`：corpus 列表，每条记录含 `citation`、`source`、`indexed_text` 等字段

**幂等性：** 已加载时直接返回，不重复读取文件。

### 4.2 `src/retrieval/rerank.py`（新建）

职责：封装 `bge-reranker-v2-m3` 的加载和批量打分。

```python
"""Cross-encoder reranker using bge-reranker-v2-m3."""

RERANKER_PATH = str(ROOT_DIR / "models" / "bge-reranker-v2-m3")
_reranker = None   # FlagReranker 实例，懒加载

def reranker_exists() -> bool:
    """检查 reranker 模型目录是否存在。"""

def _load_reranker() -> None:
    """懒加载 FlagReranker（首次调用时初始化，幂等）。"""

def rerank(
    query: str,
    candidates: list[str],
    corpus_texts: dict[str, str],
    *,
    top_k: int = 100,
    batch_size: int = 32,
) -> list[str]:
    """对 candidates[:top_k] 使用 reranker 打分并重排，返回完整的 citation 列表。
    
    参数：
        query: 拼接后的查询字符串（raw_query + " " + search_text）
        candidates: RRF 排序后的完整 citation list
        corpus_texts: citation → indexed_text 映射
        top_k: 送入 reranker 打分的候选数（默认 100）
        batch_size: reranker 批量推理大小
    
    返回：
        重排后的完整 citation list。
        candidates[:top_k] 按 reranker score 重排，
        candidates[top_k:] 按 RRF 原顺序追加到末尾（不丢弃）。
    
    缺失文本处理：
        若某个 citation 在 corpus_texts 中无对应文本，score 设为 -inf，排在所有
        可打分候选之后（仍在 candidates[top_k:] 之前）。
    """
```

**依赖库：** `FlagEmbedding`（已在环境中安装，与 bge-m3 embedding 共用）

**加载方式：**
```python
from FlagEmbedding import FlagReranker
_reranker = FlagReranker(RERANKER_PATH, use_fp16=True)
```

**打分调用：**
```python
scores = _reranker.compute_score(
    [[query, corpus_texts[citation]] for citation in to_rank],
    batch_size=batch_size,
    normalize=True,  # 输出 [0, 1] sigmoid 分数
)
```

不显式截断文本，由模型自身处理序列长度上限。

### 4.3 `src/retrieval/bm25.py` 和 `src/retrieval/dense.py`（修改）

两个模块当前都独立加载 `corpus.pkl`（内存中各一份），统一改为从 `corpus.py` 取：

**`bm25.py`**：
- `_load_index()` 中删除 `pickle.load(corpus.pkl)` 逻辑
- 改为 `from retrieval import corpus; _corpus_court = corpus.get_corpus_court(); _corpus_law = corpus.get_corpus_law()`
- BM25 index 本身（bm25s.BM25.load）仍在 `bm25.py` 内加载

**`dense.py`**：
- `_load_index()` 中同样删除 `pickle.load(corpus.pkl)` 逻辑
- 改为从 corpus 模块取 `_corpus_court` / `_corpus_law`
- FAISS index 和 SentenceTransformer 仍在 `dense.py` 内加载

### 4.4 `src/query/run.py`（修改）

**新增模块级状态：**
```python
from retrieval.rerank import reranker_exists, _load_reranker, rerank
_USE_RERANK: bool = reranker_exists()
```

**`predict_citations()` 新增参数：**
```python
def predict_citations(
    query: str,
    *,
    use_rerank: bool = True,
    rerank_top_k: int = 100,
    rerank_batch_size: int = 32,
    # ... 现有参数不变
) -> list[str]:
```

**`predict_citations()` 新增逻辑（RRF 之后）：**
```python
rrf_result = weighted_rrf(rankings, rrf_k=rrf_k)

if use_rerank and _USE_RERANK:
    rerank_query = query
    if search_text is not None:
        rerank_query = query + " " + search_text
    rrf_result = rerank(
        rerank_query,
        rrf_result,
        corpus_texts=corpus.get_corpus_texts(),
        top_k=rerank_top_k,
        batch_size=rerank_batch_size,
    )

return rrf_result[:k]
```

**新增 CLI 参数：**
```
--no-rerank              关闭 reranker（默认开启）
--rerank-top-k INT       送入 reranker 的候选数，默认 100
--rerank-batch-size INT  reranker 批量大小，默认 32
```

**启动时加载序列：**
```python
corpus.load_corpus()       # 新增（corpus 独立模块）
_load_bm25_index()
_load_dense_index(...)
if _USE_RERANK:
    _load_reranker()
```

**`_print_run_config()` 新增行：**
```
reranker:        ON  (top_k=100, batch_size=32) | OFF
```

---

## 5. 降级策略

| 条件 | 行为 |
|------|------|
| `models/bge-reranker-v2-m3` 不存在 | `_USE_RERANK = False`，跳过 reranker，退化为纯 RRF 输出 |
| `--no-rerank` 传入 | `use_rerank = False`，跳过 reranker |
| citation 在 corpus_texts 中缺失 | score = -inf，排在所有可打分候选之后 |
| 超出 `rerank_top_k` 的候选 | 按 RRF 原顺序追加到重排结果末尾，不丢弃 |

---

## 6. 数据流（完整）

```
run.py 启动
  → corpus.load_corpus()           # 加载 corpus.pkl（唯一加载点）
  → _load_bm25_index()             # bm25.py：从 corpus 模块取 citation_to_idx，加载 BM25 index
  → _load_dense_index()            # dense.py：加载 FAISS index（不变）
  → _load_reranker()               # rerank.py：加载 bge-reranker-v2-m3

per-query predict_citations(query)
  → rewrite_query(query)           # → search_text (DE)
  → retrieve_bm25_parts(...)       # → extracted, bm25_court, bm25_law
  → retrieve_dense_parts(...)      # → dense_court, dense_law
  → weighted_rrf(rankings)         # → rrf_result (全量)
  → rerank(rerank_query,           # → rrf_result 重排（前 100 精排）
           rrf_result,
           corpus_texts, top_k=100)
  → rrf_result[:k]                 # 最终截断（k 由 --k 控制）
```

---

## 7. 不在本次设计范围内

- Reranker 微调（使用 LEXam 数据训练）
- Citation Selector（LLM 选择最终引用集合）
- Score gap 自适应截断
- Citation graph expansion
