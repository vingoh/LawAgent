# Adaptive Count: n_rel23 加权项设计

**日期：** 2026-06-21
**状态：** 已实现
**位置：** `src/retrieval/selector.py` — `adaptive_count()`

---

## 1. 背景

当前 `adaptive_count()` 使用两项加权公式决定最终 citation 数量：

```python
n_final = clamp(round(4/7 * n_LLM + 3/7 * n_elbow), min_val, max_val)
```

- `n_LLM`：LLM verifier 输出的 `estimated_answer_count`（问题复杂度全局估计）
- `n_elbow`：`final_score` 降序曲线上的 Kneedle 拐点

LLM verifier 还会对每个候选打 `relevance` 分（0–3），目前仅用于 `fuse_scores()` 的 boost，未参与 citation 数量计算。

**问题：** 全局估计 `n_LLM` 与逐条 relevance 判定可能不一致。例如 LLM 估计需要 5 条 citation，但实际给 12 个候选打了 relevance 2/3，现有公式无法反映这一信号。

---

## 2. 目标

在 `adaptive_count()` 中增加第三项 `n_rel23`（relevance 为 2 或 3 的候选数量），以 0.2 / 0.4 / 0.4 权重与 `n_LLM`、`n_elbow` 加权融合。

---

## 3. 新公式

```python
n_rel23 = sum(1 for c in candidates if c.relevance in (2, 3))

n_final = clamp(
    round(0.2 * n_LLM + 0.4 * n_elbow + 0.4 * n_rel23),
    min_val=3,
    max_val=40,
)
```

### 权重映射

| 信号 | 权重 | 含义 |
|------|------|------|
| `n_LLM` | 0.2 | LLM 对问题复杂度的全局估计 |
| `n_elbow` | 0.4 | `final_score` 曲线的结构拐点 |
| `n_rel23` | 0.4 | LLM 逐条判定为相关（2/3）的候选数 |

### 与旧公式对比

| | 旧公式 | 新公式 |
|---|--------|--------|
| `n_LLM` 权重 | 4/7 ≈ 0.57 | 0.2 |
| `n_elbow` 权重 | 3/7 ≈ 0.43 | 0.4 |
| `n_rel23` 权重 | — | 0.4 |

全局估计权重显著降低，逐条 relevance 计数与 elbow 信号权重相等。

---

## 4. `n_rel23` 定义

- **统计范围：** 全部 `candidates` 列表（verifier 仅对 top `verifier_top_k`（默认 60）打分，其余 `relevance=None`）
- **计入条件：** `c.relevance in (2, 3)`
- **不计入：** `c.relevance in (0, 1)` 或 `c.relevance is None`

relevance 等级含义（与现有定义一致）：

| 值 | 含义 |
|----|------|
| 3 | 直接回答法律问题 |
| 2 | 有用的支持性权威 |
| 1 | 弱相关 / 背景性 |
| 0 | 无关或误导性 |

---

## 5. 实现方案

**采用方案：** 在 `adaptive_count()` 内部计算 `n_rel23`，不修改函数签名或调用方。

```python
def adaptive_count(
    candidates: list[Candidate],
    n_llm: int,
    *,
    min_val: int = 3,
    max_val: int = 40,
) -> int:
    """Compute n_final from n_LLM, n_elbow, and n_rel23."""
    scores = [c.final_score for c in candidates]
    n_elbow = _elbow(scores)
    n_rel23 = sum(1 for c in candidates if c.relevance in (2, 3))
    raw = round(0.2 * n_llm + 0.4 * n_elbow + 0.4 * n_rel23)
    return max(min_val, min(max_val, raw))
```

`run_selector()`、`assemble()`、CLI 参数均不变。

---

## 6. 降级行为

| 场景 | `n_rel23` | 实际公式 |
|------|-----------|----------|
| verifier 正常 | 实际计数 | `0.2·n_LLM + 0.4·n_elbow + 0.4·n_rel23` |
| verifier 失败（两次重试后） | 0（全部 `relevance=None`） | `0.2·10 + 0.4·n_elbow` |
| `--no-llm-verify` | 0 | `0.2·10 + 0.4·n_elbow` |

verifier 跳过时公式自动退化为两项加权，无需额外分支逻辑。

---

## 7. 改动范围

| 文件 | 变更 |
|------|------|
| `src/retrieval/selector.py` | 修改 `adaptive_count()` 公式和 docstring |
| `tests/retrieval/test_selector.py` | 更新 `test_adaptive_count_formula`；新增 `n_rel23` 相关测试 |
| `docs/superpowers/specs/2026-06-21-selector-pipeline-design.md` | 更新 §4.4 Adaptive Citation Count 公式 |

---

## 8. 测试

### 8.1 公式验证（mock `_elbow`）

```python
# n_rel23 被计入
candidates = [_make_candidate(f"C{i}", relevance=3 if i < 5 else 0) for i in range(10)]
with patch.object(sel, "_elbow", return_value=8):
    n = adaptive_count(candidates, n_llm=7, min_val=3, max_val=40)
# expected = round(0.2*7 + 0.4*8 + 0.4*5) = round(6.6) = 7
assert n == 7
```

### 8.2 relevance=None 不计入

```python
candidates = [_make_candidate(f"C{i}") for i in range(10)]  # all relevance=None
with patch.object(sel, "_elbow", return_value=8):
    n = adaptive_count(candidates, n_llm=7, min_val=3, max_val=40)
# expected = round(0.2*7 + 0.4*8 + 0.4*0) = round(4.6) = 5
assert n == 5
```

### 8.3 仅 relevance=2 计入

```python
candidates = [_make_candidate(f"C{i}", relevance=2 if i < 3 else 1) for i in range(10)]
with patch.object(sel, "_elbow", return_value=5):
    n = adaptive_count(candidates, n_llm=10, min_val=3, max_val=40)
# n_rel23=3; expected = round(0.2*10 + 0.4*5 + 0.4*3) = round(4.2) = 4
assert n == 4
```

### 8.4 min/max clamp 仍生效

现有 `test_adaptive_count_respects_min` 和 `test_adaptive_count_respects_max` 继续有效，无需修改逻辑。

---

## 9. 不在本次范围内

- 权重调参或 CLI 暴露权重参数
- 修改 `fuse_scores()` 的 relevance boost 表
- 修改 LLM verifier prompt 或 `estimated_answer_count` 逻辑
