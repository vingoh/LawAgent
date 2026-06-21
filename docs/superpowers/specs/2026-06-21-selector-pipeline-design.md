# Selector Pipeline 设计文档

**日期：** 2026-06-21
**状态：** 待实现
**位置：** `src/retrieval/selector.py`（新建）

---

## 1. 背景

当前 pipeline 在 reranker 之后直接截断输出 top-k citations，存在以下问题：

1. **固定截断**：`k` 参数不随 query 复杂度自适应
2. **reranker 误排**：排名靠后但 query 中直接引用或同 parent case 的 citation 可能被丢弃
3. **无语义验证**：reranker 是 cross-encoder，无法判断"法律问题维度的相关性"
4. **长尾遗漏**：reranker top 50 可能完全由某一路检索主导，BM25 精确命中的长尾候选丢失

Selector pipeline 在 reranker 之后引入 5 个有序阶段，解决上述问题。

---

## 2. 整体架构

### 2.1 修改后的 pipeline

```
英文 query (CSV)
  │
  ├─► rewrite_query()                 → query_info (RewriteResult)
  │
  ├─► retrieve_bm25_parts()           → extracted, bm25_court, bm25_law
  ├─► retrieve_dense_parts()          → dense_court, dense_law
  │
  ├─► weighted_rrf()                  → rrf_result (全量) + rrf_scores dict
  │                                     ↑ rrf.py 修改：同时返回 score dict
  │
  ├─► rerank_with_scores(top_k=100)   → [(citation, rerank_score), ...]
  │                                     ↑ rerank.py 新增函数
  │
  ├─► build_candidates()              → list[Candidate]（挂载全部元数据）
  │
  └─► run_selector()                  → list[str]（最终 citations）
        │
        ├── cheap_expand()            确定性扩展：same_parent / same_code / rescue
        ├── llm_verify()              LLM 打 relevance 分 + 估计 n_LLM
        ├── fuse_scores()             final_score = rerank_score + llm_boost + rule_boost
        ├── adaptive_count()          n_final = clamp(round(4/7·n_LLM + 3/7·n_elbow), 3, 20)
        └── assemble()                约束选取 + 输出 list[str]
```

### 2.2 新增 / 修改文件

| 文件 | 变更 |
|------|------|
| `src/retrieval/selector.py` | **新建**：selector pipeline 全部逻辑（`Candidate` dataclass + 5 个阶段函数 + `run_selector()` 入口） |
| `src/retrieval/rerank.py` | 新增 `rerank_with_scores()`；原 `rerank()` 内部调用它，保持向后兼容 |
| `src/retrieval/rrf.py` | 修改 `weighted_rrf()` 同时返回 score dict（`tuple[list[str], dict[str, float]]`） |
| `src/query/run.py` | 调用 `rerank_with_scores()` → `build_candidates()` → `run_selector()`；新增 `--no-select` / `--verifier-top-k` CLI 参数 |

---

## 3. 数据结构

### 3.1 `Candidate` dataclass

```python
@dataclass
class Candidate:
    # ── 基础信息 ──────────────────────────────────────────
    citation: str
    rerank_score: float          # bge-reranker sigmoid score，[0, 1]
    rrf_score: float             # weighted_rrf score

    # ── 元数据（build_candidates 阶段填充）────────────────
    source_set: frozenset[str]   # 命中的检索路：
                                 #   "extracted" / "bm25_court" / "bm25_law"
                                 #   / "dense_court" / "dense_law"
    direct_regex_hit: bool       # citation 在 extracted channel 里（query 中直接出现）
    expected_code_match: bool    # citation 的 law code 在 query_info.expected_codes 里
    expansion_type: str | None   # None（原始候选）/ "same_parent" / "same_code"
                                 #   / "extracted_rescue" / "bm25_court_rescue" 等

    # ── LLM verifier 输出（verifier 跳过时为 None）────────
    relevance: int | None = None       # 0-3（见定义）
    llm_reason: str | None = None

    # ── score fusion 输出 ─────────────────────────────────
    final_score: float = 0.0
```

### 3.2 `relevance` 定义

| 值 | 含义 |
|----|------|
| 3 | 直接回答法律问题（directly answers the legal issue） |
| 2 | 有用的支持性权威（useful supporting authority） |
| 1 | 弱相关 / 背景性（weak / background relevance） |
| 0 | 无关或误导性（unrelated or misleading） |

### 3.3 `build_candidates()` 职责

接收 reranker top 100 的 scored list 及各路检索结果，构建 `Candidate` 列表：

```python
def build_candidates(
    scored: list[tuple[str, float]],       # rerank_with_scores() 输出
    rrf_scores: dict[str, float],          # weighted_rrf 返回的 score dict
    source_rankings: dict[str, list[str]], # 各路原始排名
    query_info,                            # RewriteResult
) -> list[Candidate]:
```

- `source_set`：遍历 `source_rankings`，记录该 citation 在哪些路里出现
- `direct_regex_hit`：`citation in set(source_rankings["extracted"])`
- `expected_code_match`：从 citation 中提取 law code，判断是否在 `query_info.expected_codes` 里

---

## 4. 阶段设计

### 4.1 Cheap Expansion

**目的：** 在不调用 LLM 的情况下，从 reranker top 100 里补回可能被遗漏的相关候选。

#### Seed 选择（启发式）

```python
def _is_seed(c: Candidate, seed_score_threshold: float = 0.5) -> bool:
    return c.rerank_score >= seed_score_threshold or c.direct_regex_hit
```

#### 扩展类型 1：same_parent case expansion

解析 BGE 判例的 parent（去掉 Erwägung 部分）：

```
"BGE 145 IV 154 E. 1.1"  →  parent: "BGE 145 IV 154"
```

正则：`^(BGE\s+\d+\s+(?:I{1,4}|IV|V)\s+\d+)(?:\s+E\..*)?$`

- 收集所有 seed 的 parent
- 在 reranker top 100 里找同 parent 的**其他**候选（seed 之外的）
- 对这些候选设 `expansion_type = "same_parent"`（打标记，不新增候选——它们已在 top 100 pool 里）
- "最多保留 3 个"是 **assemble 阶段**的约束（Section 4.5），expand 阶段对所有匹配都打标记

#### 扩展类型 2：same_code expansion

提取 law code：`\b(StPO|StGB|OR|ZGB|ZPO|BGG|BV|ATSG|IVG|AHVG|UVG|AsylG|AIG|SchKG|DSG)\b`

- 收集所有 seed 的 code
- 在 reranker top 100 里找同 code 的其他候选（seed 之外的）
- 对这些候选设 `expansion_type = "same_code"`（打标记，不新增候选）
- "最多保留 5 个"同理为 assemble 阶段约束

#### 扩展类型 3：direct_regex_hit keep

所有 `direct_regex_hit = True` 的候选强制进入 merged pool，`expansion_type` 保持 `None`（它们本来就是原始候选，不是新增扩展）。

#### 扩展类型 4：source diversity rescue

防止某一路检索主导导致长尾精确命中丢失：

```python
RESCUE_SOURCES = ["extracted", "bm25_court", "bm25_law", "dense_court", "dense_law"]
RESCUE_TOP_K = 10  # 每路各取前 10
```

对每路取 `source_rankings[src][:rescue_top_k]`，找出**不在 reranker top 100 pool 里**的候选，将其作为新候选补入 pool：
- `expansion_type = f"{src}_rescue"`
- `rerank_score = 0.0`（从未经过 reranker 打分）
- `rrf_score` 从 rrf_scores dict 取（这些候选参与了 RRF 融合，score 存在）
- `direct_regex_hit`、`expected_code_match`、`source_set` 正常填充

已在 reranker top 100 里的候选不重复添加（也不覆盖其 expansion_type）。

#### 函数签名

```python
def cheap_expand(
    candidates: list[Candidate],
    source_rankings: dict[str, list[str]],
    rrf_scores: dict[str, float],
    *,
    max_per_parent: int = 3,
    max_per_code: int = 5,
    rescue_top_k: int = 10,
    seed_score_threshold: float = 0.5,
) -> list[Candidate]:
```

扩展后候选池规模：
- same_parent/same_code：不新增候选（仅打标记），pool 仍为 100
- rescue：最多新增 `5 路 × 10 = 50` 个（大多数已在 top 100 里，实际新增约 10–30）
- 扩展后 pool ≈ **100–130 个候选**

---

### 4.2 LLM Verifier

**目的：** 对扩展后候选池打语义相关分，估计最终应输出 citation 数量。

#### 输入

取扩展后候选池按 `rerank_score` 降序的 **top `verifier_top_k`（默认 60）** 个，防止 token 超限。

每个候选传给 LLM 的字段：

```json
{
  "id": "C001",
  "citation": "BGE 145 IV 154 E. 3.2",
  "text": "...corpus text 前 300 字符...",
  "rerank_score": 0.87,
  "direct_hit": false
}
```

#### Prompt context

```json
{
  "question": "Can a person be convicted of fraud if...",
  "legal_issue": "Betrug durch arglistige Täuschung",
  "expected_codes": ["StGB", "OR"],
  "expected_articles": ["Art. 146 StGB"],
  "candidates": [...]
}
```

#### 输出 JSON schema

```json
{
  "candidate_scores": [
    {
      "id": "C001",
      "relevance": 3,
      "reason": "Directly addresses the fraud elements under Art. 146 StGB"
    }
  ],
  "estimated_answer_count": 5
}
```

**约束（写入 system prompt）：**
- 只能使用输入中出现的 `id`，不允许生成新 id
- 每个输入候选必须给出评分
- `estimated_answer_count` 基于 query 的法律问题复杂度估计

#### 失败处理

1. JSON 解析失败 / 调用异常 → 重试一次
2. 仍失败 → `verifier_skipped = True`，全部 `relevance = None`，`n_LLM = 10`（默认先验）

#### 函数签名

```python
def llm_verify(
    candidates: list[Candidate],
    query: str,                  # 原始英文 query
    query_info,                  # RewriteResult
    corpus_texts: dict[str, str],
    *,
    verifier_top_k: int = 60,
    text_snippet_len: int = 300,
) -> tuple[list[Candidate], int]:   # (candidates with relevance filled, n_LLM)
```

---

### 4.3 Final Score Fusion

**公式：**

```
final_score = rerank_score + llm_relevance_boost + rule_boost
```

#### LLM relevance boost

| relevance | boost |
|-----------|-------|
| 3 | +0.35 |
| 2 | +0.15 |
| 1 | -0.10 |
| 0 | -0.40 |
| None（verifier 跳过） | 0.0 |

#### Rule boost

| 规则 | boost | 说明 |
|------|-------|------|
| `direct_regex_hit` | +0.25 | query 中直接引用的 citation，强信号 |
| `expected_code_match` | +0.06 | citation 的 law code 在 expected_codes 里 |
| `len(source_set) >= 2` | +0.04 | 多路检索均命中，稳定性高 |
| `expansion_type == "same_parent"` | +0.03 | 同判例扩展 |
| `expansion_type == "same_code"` | +0.02 | 同法典扩展 |
| `expansion_type` ends with `"_rescue"` | -0.05 | rescue 候选惩罚（-0.03 偏轻，-0.05 更能将其推到截断线以下） |
| `direct_regex_hit AND expansion_type is not None` | -0.05 | 直接命中但位于 reranker top 100 较低位置，抵消部分 direct_hit 红利 |

Rule boost 可以叠加（`direct_regex_hit` + `multi_source` 等）。

#### 函数签名

```python
_LLM_BOOST: dict[int, float] = {0: -0.40, 1: -0.10, 2: 0.15, 3: 0.35}

def fuse_scores(candidates: list[Candidate]) -> list[Candidate]:
    """原地计算每个 Candidate 的 final_score，返回按 final_score 降序排列的列表。"""
```

---

### 4.4 Adaptive Citation Count

**公式：**（详见 `2026-06-21-adaptive-count-rel23-design.md`）

```python
n_rel23 = sum(1 for c in candidates if c.relevance in (2, 3))

n_final = clamp(
    round(0.2 * n_LLM + 0.4 * n_elbow + 0.4 * n_rel23),
    min_val=3,
    max_val=40,
)
```

`n_rel23`：LLM verifier 判定 relevance 为 2 或 3 的候选数量。

#### n_elbow 计算

使用 Kneedle 算法（`kneed` 库）：在 final_score 降序曲线上，找到距离"首尾连线"（min-max line）垂直距离最远的点。

```python
from kneed import KneeLocator

def _elbow(scores: list[float]) -> int:
    """返回 elbow 点之前的候选数量（Kneedle 算法）。"""
    if len(scores) <= 2:
        return len(scores)
    x = list(range(len(scores)))
    knl = KneeLocator(x, scores, curve="convex", direction="decreasing")
    knee = knl.knee
    return (knee + 1) if knee is not None else len(scores)
```

若 `KneeLocator` 未检测到 knee（曲线过于平滑），`knl.knee` 为 `None`，回退为全部候选数量，由 `clamp` 上限（20）兜底。

依赖：`kneed>=0.8`（已安装）。

#### verifier 跳过时

`n_LLM = 10`（默认先验），`n_rel23 = 0`，公式退化为 `0.2·10 + 0.4·n_elbow`。

#### 函数签名

```python
def adaptive_count(
    candidates: list[Candidate],   # 已按 final_score 降序排列
    n_llm: int,
) -> int:                          # n_final
```

---

### 4.5 Final Assembly

**约束执行顺序：**

1. 候选按 `final_score` 降序排列（`fuse_scores` 已完成）
2. 去重（citation 字符串完全一致）
3. 按 BGE parent 约束：同一 BGE parent 最多保留 `max_per_parent = 3` 个（cheap_expand 标记的 same_parent 候选在此被限制）
4. 取 score 排名前 `n_final` 个作为主体
5. 强制保留所有 `direct_regex_hit` 候选（若已在主体内则不重复）
6. 移除 `rerank_score < rescue_score_floor`（默认 0.1）的 rescue 候选（不强行填满 n_final）
7. **不允许** LLM 生成 citation——所有输出只能来自 `Candidate.citation` 字段

**`direct_regex_hit` 强制保留说明：**

若 `direct_regex_hit` 候选在 top `n_final` 截断线之外，追加到结果末尾（不占用 n_final 计数，独立追加）。这种情况应较少出现（`direct_regex_hit` 有 +0.25 boost，理论上会进入主体）。

#### 函数签名

```python
def assemble(
    candidates: list[Candidate],   # 已按 final_score 降序排列
    n_final: int,
    *,
    max_per_parent: int = 3,
    rescue_score_floor: float = 0.1,
) -> list[str]:                    # 最终 citation list
```

---

## 5. `run_selector()` 入口

```python
def run_selector(
    query: str,
    query_info,                          # RewriteResult（可为 None，当 rewrite 关闭时）
    scored: list[tuple[str, float]],     # rerank_with_scores() 输出
    rrf_scores: dict[str, float],
    source_rankings: dict[str, list[str]],
    corpus_texts: dict[str, str],
    *,
    # cheap_expand 参数
    max_per_parent: int = 3,
    max_per_code: int = 5,
    rescue_top_k: int = 10,
    seed_score_threshold: float = 0.5,
    # llm_verify 参数
    use_llm_verify: bool = True,
    verifier_top_k: int = 60,
    text_snippet_len: int = 300,
    # adaptive_count 参数
    min_citations: int = 3,
    max_citations: int = 40,
    # assemble 参数
    max_per_parent_final: int = 3,
    rescue_score_floor: float = 0.1,
) -> list[str]:
    candidates = build_candidates(scored, rrf_scores, source_rankings, query_info)
    candidates = cheap_expand(candidates, source_rankings, rrf_scores, ...)
    n_llm = 10  # default
    if use_llm_verify and query_info is not None:
        candidates, n_llm = llm_verify(candidates, query, query_info, corpus_texts, ...)
    candidates = fuse_scores(candidates)
    n_final = adaptive_count(candidates, n_llm, min_val=min_citations, max_val=max_citations)
    return assemble(candidates, n_final, ...)
```

---

## 6. `run.py` 集成

### 修改点

**`predict_citations()` 调用链变化（reranker 之后）：**

```python
# 现在
rrf_result = rerank(rerank_query, rrf_result, corpus_texts, top_k=rerank_top_k, ...)
return rrf_result[:k]

# 修改后
rrf_result, rrf_scores = weighted_rrf(rankings, rrf_k=rrf_k)   # rrf.py 改动
scored = rerank_with_scores(rerank_query, rrf_result, corpus_texts, top_k=rerank_top_k, ...)
source_rankings = {"extracted": extracted, "bm25_court": bm25_court, ...}

if use_select and selector_exists():
    return run_selector(query, rewrite_result, scored, rrf_scores, source_rankings, corpus_texts, ...)
else:
    return [cit for cit, _ in scored][:k]
```

### 新增 CLI 参数

```
--no-select              关闭 selector pipeline（默认开启，若 LLM 可用）
--verifier-top-k INT     LLM verifier 看到的候选数，默认 60
--no-llm-verify          关闭 LLM verifier（仅做 cheap expand + score fusion + 自适应截断）
```

---

## 7. 降级策略

| 条件 | 行为 |
|------|------|
| `--no-select` | 跳过整个 selector，退化为 `reranked[:k]` |
| `--no-llm-verify` | 仅执行 cheap_expand + fuse_scores（无 llm_boost）+ adaptive_count + assemble |
| LLM verifier 重试后仍失败 | `relevance = None`，`n_LLM = 10`，其余阶段正常执行 |
| `query_info is None`（`--no-rewrite`） | verifier 跳过；expected_code_match 全为 False |
| rescue 候选 `rerank_score < 0.1` | assemble 阶段移除，不强行填满 n_final |

---

## 8. 不在本次设计范围内

- LLM verifier 的 system prompt 微调（留给实现阶段迭代）
- Reranker 微调（LEXam 数据训练）
- Citation graph expansion（超出当前 RRF candidate pool 的外部扩展）
- 评估 selector 对 Macro F1 的实际提升（需跑 val.csv 对比）
