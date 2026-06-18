---
name: Build Index Plan
overview: 为瑞士法律检索系统建立 BM25 索引和评估脚本，作为 Stage 1 Baseline 的核心基础。先完成 BM25 索引，dense embedding 索引延后（需 GPU）。
todos:
  - id: req
    content: 创建 requirements.txt，安装 bm25s/pandas/numpy/tqdm
    status: completed
  - id: corpus
    content: 编写 src/indexing/build_corpus.py：合并两个 CSV，去重合并重复 citation，保存 indexes/corpus.pkl
    status: completed
  - id: bm25
    content: 编写 src/indexing/build_bm25.py：用 bm25s 构建 BM25 索引，保存到 indexes/bm25/
    status: completed
  - id: eval
    content: 编写 src/eval/macro_f1.py：Macro F1 评估脚本，支持 --k 参数，输出 F1 + Recall@k
    status: completed
  - id: run
    content: 运行完整 pipeline，记录 val 上的 Macro F1 和 Recall@50/100 基线数据
    status: in_progress
isProject: false
---

# 建立检索索引 Plan

## 数据确认

两个语料文件均已就绪：
- `dataset/court_considerations.csv` — 2,476,315 行，列：`citation`, `text`（平均 1176 字符）
- `dataset/laws_de.csv` — 175,933 行，列：`citation`, `text`, `title`

关键特性：
- laws_de 中约 74.9% 的 citation 使用标准德文缩写（OR/ZGB/StPO/...），与 gold citation 格式一致；剩余 ~25.1% 使用数字 SR 编号（如 `Art. 1 112`），这些无法匹配任何 gold citation，**需在 build_corpus 阶段过滤掉**
- court_considerations 中约 3.5% 的 citation 存在重复（同一 citation key 多段文本），需要处理
- court_considerations 中存在少量 date-embedded 的畸形 citation（如 `BGE 139 I 2 E. 1.12.2011`），需过滤
- court_considerations 包含 BGE 和非 BGE 判决（`1B_xxx/yyyy E. z.z` 等），两类均有覆盖
- val/test query 为英文，但语料为德/法/意文——**BM25 存在跨语言匹配问题**（见步骤三说明）

## 目录结构

```
LawAgent/
├── src/
│   ├── indexing/
│   │   ├── build_corpus.py   # 加载两个 CSV，去重合并 → corpus.pkl
│   │   └── build_bm25.py     # 从 corpus.pkl 构建 BM25 index
│   └── eval/
│       └── macro_f1.py       # 本地 Macro F1 评估（与 Kaggle 对齐）
├── indexes/
│   ├── bm25/                 # bm25s 保存目录
│   └── corpus.pkl            # citation 列表 + 类型元数据
└── requirements.txt
```

## 步骤一：安装依赖

依赖安装在 `condaagent` conda 环境下（不创建 requirements.txt，直接 conda/pip 安装）：

```bash
conda run -n condaagent pip install bm25s tqdm
```

`pandas` 和 `numpy` 已在该环境中安装，无需重复安装。

Dense embedding 相关（faiss-cpu, sentence-transformers）延后添加。

## 步骤二：build_corpus.py

合并两个 CSV 为统一语料，完成以下处理：

**laws_de 过滤：**
```python
import re
def has_numeric_sr_code(citation: str) -> bool:
    # 过滤掉 "Art. 1 112" 这类数字 SR 编号的条文（无法匹配 gold citation）
    return bool(re.search(r'\s[\d.]+$', citation))

laws_rows = [r for r in laws_rows if not has_numeric_sr_code(r["citation"])]
```

**court_considerations 处理：**
```python
def is_malformed_consideration(citation: str) -> bool:
    # 过滤 date-embedded 畸形 citation，如 "BGE 139 I 2 E. 1.12.2011"
    m = re.search(r'E\.\s*([\d.]+)$', citation)
    if m:
        parts = m.group(1).split('.')
        return any(len(p) == 4 for p in parts)  # 4位数字 = 年份
    return False

# 重复 citation：取最长单段文本（避免拼接导致的 BM25 长度惩罚）
grouped = {}
for row in court_rows:
    cit = row["citation"]
    if cit not in grouped or len(row["text"]) > len(grouped[cit]["text"]):
        grouped[cit] = row
```

**统一 corpus 输出：**
```python
# 最终输出: List[{"citation": str, "text": str, "source": "court"|"law"}]
# 保存为 indexes/corpus.pkl
# 同时保存 indexes/citation_to_idx.pkl (citation → corpus index 的快速查找 dict)
```

- court_considerations：indexed text = `citation + " " + text`（citation 前置，提升精确命中 TF 权重）
- laws_de：indexed text = `citation + " " + title + " " + text`

## 步骤三：build_bm25.py

使用 `bm25s` 构建索引：

```python
import bm25s, os

os.makedirs("indexes/bm25", exist_ok=True)

# 分词：小写 + 空格分词（德文不做 stemming，保留法律缩写）
indexed_texts = [doc["indexed_text"] for doc in corpus]  # citation + title + text
tokenized = bm25s.tokenize(indexed_texts)
retriever = bm25s.BM25()
retriever.index(tokenized)
retriever.save("indexes/bm25")
```

估算：
- 内存峰值约 4-8 GB（德文词汇量大，sparse matrix 较大）
- 构建时间约 15-40 分钟（单进程，CPU）

#### 跨语言问题说明

val/test query 为英文，BM25 对德文语料召回率低。Baseline 阶段采用两个补丁策略（无需翻译）：

**策略 A（citation 正则提取）：** 许多 val/test query 正文中直接嵌入了目标 citation（如 `"Art. 221 Abs. 1 lit. b StPO"`），可通过 regex 直接提取，加入最终候选：
```python
STATUTE_RE = re.compile(r'Art\.\s*\d+(?:\s+Abs\.\s*\d+)?(?:\s+lit\.\s*\w+)?\s+[A-Z][A-Za-z]+')
BGE_RE = re.compile(r'BGE\s+\d+\s+[IVX]+\s+\d+(?:\s+E\.\s*[\d.]+)?')
BGER_RE = re.compile(r'\d[A-Z]_\d+/\d{4}(?:\s+E\.\s*[\d.]+)?')
```

**策略 B（延后）：** 将英文 query 翻译为德文后再做 BM25 检索，在 Stage 2+ 实现。

## 步骤四：macro_f1.py

评估脚本，与 Kaggle 评分逻辑完全对齐：

```python
def compute_macro_f1(gold_list: list[set], pred_list: list[set]) -> float:
    # 每条 query 精确字符串匹配计算 F1，取平均

def compute_recall_at_k(gold: set, candidates: list[str], k: int) -> float:
    # candidates 是 BM25 返回的有序列表（未截断），取前 k 条计算 Recall
    # Recall@k = |gold ∩ candidates[:k]| / |gold|

def retrieve_bm25(query: str, k: int = 200) -> list[str]:
    # BM25 检索 top-k（有序），再合并 citation regex 直接提取结果
    # 返回有序 citation 列表（供不同 k 截断复用，无需多次检索）

def evaluate_val(top_k: int = 100):
    # 对每条 query，检索一次返回 top-200 有序候选
    # 输出：
    # 1. 逐条 query 明细：query_id, gold_count, pred_count, tp, P, R@50, R@100, R@200, F1@top_k, missed_citations
    # 2. 聚合行：Macro F1@top_k + 平均 Recall@50/100/200
```

输出示例（`--k 100`）：
```
query_id  | gold | tp@100 | R@50  | R@100 | R@200 | F1@100 | missed (first 3)
val_001   |  42  |   8    | 0.14  | 0.19  | 0.26  |  0.11  | [BGE 132 I 21 E. 3.2, ...]
val_002   |  17  |   3    | 0.12  | 0.18  | 0.24  |  0.05  | [Art. 8 Abs. 1 ATSG, ...]
...
AGGREGATE |  -   |   -    | 0.13  | 0.19  | 0.25  |  0.08  |
```

Recall@k 的意义：衡量召回阶段的**理论上限**——若 Recall@200 < 0.8，后续 Reranker 和 Selector 无论多好，最终 F1 也有硬性天花板。

运行方式：

```bash
conda run -n condaagent python src/eval/macro_f1.py --k 50 100 200
```

## 实现顺序

1. `conda run -n condaagent pip install bm25s tqdm` — 安装依赖
2. `src/indexing/build_corpus.py` — 加载、过滤（数字 SR 码、畸形 citation）、去重、合并语料，生成 `indexes/corpus.pkl` + `indexes/citation_to_idx.pkl`
3. `src/indexing/build_bm25.py` — 构建 BM25 索引（含 `os.makedirs`），保存到 `indexes/bm25/`
4. `src/eval/macro_f1.py` — 评估脚本（默认 k=100，输出逐条 query 明细 + 聚合 Macro F1）
5. 在 val 10 条 query 上运行（BM25 + citation regex 提取），记录逐条 F1 和 Recall@50/100/200

## 延后（不在本 Plan 范围）

- Dense embedding 索引（需 GPU，时间 >10 小时）
- Query rewriting
- Reranker
