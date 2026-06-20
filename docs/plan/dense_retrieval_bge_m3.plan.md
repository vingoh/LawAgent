# Dense Retrieval (bge-m3 + FAISS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add BAAI/bge-m3 dense vector retrieval as two independent ranking signals (court + law) fused alongside existing BM25 via 5-way weighted RRF.

**Architecture:** `build_dense.py` encodes the corpus in 50 K-doc chunks (float16, checkpoint-resumable) and merges into FAISS IndexFlatIP indexes. `dense.py` lazy-loads the indexes and exposes `retrieve_dense_parts()`. `run.py` reads per-source existence flags at startup and adds dense signals to the RRF rankings list.

**Tech Stack:** sentence-transformers (bge-m3 on MPS), faiss-cpu, numpy, existing `corpus.pkl` + `weighted_rrf()`.

**Spec:** `docs/superpowers/specs/2026-06-19-dense-retrieval-design.md`

---

## Task 0: Environment Setup

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Install new dependencies**

```bash
conda activate agent
pip install sentence-transformers faiss-cpu
```

Verify:
```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python -c "import sentence_transformers, faiss; print('OK')"
```
Expected output: `OK`

- [ ] **Step 2: Add `models/` to `.gitignore`**

Open `.gitignore` and append:
```
models/
```

Final `.gitignore` should contain these lines (among others):
```
dataset/
indexes**/
models/
results/
```

- [ ] **Step 3: Commit setup**

```bash
git add .gitignore
git commit -m "chore: add sentence-transformers, faiss-cpu; gitignore models/"
```

---

## Task 1: `src/indexing/build_dense.py`

**Files:**
- Create: `src/indexing/build_dense.py`

- [ ] **Step 1: Create the file with full implementation**

```python
"""
Build dense FAISS indexes from corpus.pkl using BAAI/bge-m3.

Inputs:  indexes/corpus.pkl
         models/bge-m3/          (manually downloaded)
Outputs: indexes/dense_law/index.faiss
         indexes/dense_court/index.faiss

Usage:
    conda run -n agent python src/indexing/build_dense.py --source law
    conda run -n agent python src/indexing/build_dense.py --source court
    conda run -n agent python src/indexing/build_dense.py
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT_DIR    = Path(__file__).resolve().parents[2]
INDEX_DIR   = ROOT_DIR / "indexes"
CORPUS_PATH = INDEX_DIR / "corpus.pkl"
MODEL_PATH  = str(ROOT_DIR / "models" / "bge-m3")

CHUNK_SIZE        = 50_000
ENCODE_BATCH_SIZE = 16
DIM               = 1024


def _encode_source(
    docs: list[dict],
    source: str,
    model: SentenceTransformer,
) -> None:
    output_dir = INDEX_DIR / f"dense_{source}"
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.faiss"

    if index_path.exists():
        print(f"[{source}] index.faiss already exists — skipping.", file=sys.stderr)
        return

    n = len(docs)
    n_chunks = (n + CHUNK_SIZE - 1) // CHUNK_SIZE
    print(f"[{source}] {n:,} docs → {n_chunks} chunks of {CHUNK_SIZE:,}", file=sys.stderr)

    # ── Encode phase ──────────────────────────────────────────────────────────
    for chunk_idx, start in enumerate(range(0, n, CHUNK_SIZE)):
        chunk_path = chunks_dir / f"{chunk_idx:04d}.npy"
        if chunk_path.exists():
            print(
                f"[{source}] chunk {chunk_idx}/{n_chunks - 1} exists, skipping",
                file=sys.stderr,
            )
            continue

        batch = docs[start : start + CHUNK_SIZE]
        texts = [d["indexed_text"] for d in batch]
        print(
            f"[{source}] encoding chunk {chunk_idx}/{n_chunks - 1} ({len(texts):,} docs) …",
            file=sys.stderr,
        )
        embs = model.encode(
            texts,
            batch_size=ENCODE_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float16")

        # Atomic write: .npy.tmp → .npy so partial crashes don't look like done chunks
        tmp_path = chunk_path.with_suffix(".npy.tmp")
        np.save(str(tmp_path), embs)
        os.rename(tmp_path, chunk_path)
        print(f"[{source}] chunk {chunk_idx} saved → {chunk_path}", file=sys.stderr)

    # ── Merge phase ───────────────────────────────────────────────────────────
    chunk_files = sorted(chunks_dir.glob("*.npy"))
    print(
        f"[{source}] merging {len(chunk_files)} chunks into FAISS IndexFlatIP …",
        file=sys.stderr,
    )
    index = faiss.IndexFlatIP(DIM)
    for cp in chunk_files:
        chunk_embs = np.load(str(cp)).astype("float32")
        index.add(chunk_embs)

    faiss.write_index(index, str(index_path))
    print(
        f"[{source}] index saved → {index_path}  ({index.ntotal:,} vectors)",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build dense FAISS indexes")
    parser.add_argument(
        "--source",
        choices=["court", "law", "all"],
        default="all",
        help="Which source to index (default: all). Build 'law' first to validate.",
    )
    args = parser.parse_args()

    print("Loading corpus.pkl …", file=sys.stderr)
    with open(CORPUS_PATH, "rb") as f:
        corpus: list[dict] = pickle.load(f)
    court_docs = [d for d in corpus if d["source"] == "court"]
    law_docs   = [d for d in corpus if d["source"] == "law"]
    print(f"court: {len(court_docs):,}  law: {len(law_docs):,}", file=sys.stderr)

    print(f"Loading model from {MODEL_PATH} …", file=sys.stderr)
    model = SentenceTransformer(MODEL_PATH, device="mps")

    if args.source in ("law", "all"):
        _encode_source(law_docs, "law", model)
    if args.source in ("court", "all"):
        _encode_source(court_docs, "court", model)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import is clean (no runtime errors at import time)**

```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python -c "import sys; sys.path.insert(0,'src/indexing'); import build_dense; print('import OK')"
```
Expected: `import OK`

- [ ] **Step 3: Commit**

```bash
git add src/indexing/build_dense.py
git commit -m "feat: add build_dense.py — chunked bge-m3 encode + FAISS IndexFlatIP builder"
```

---

## Task 2: `src/retrieval/dense.py`

**Files:**
- Create: `src/retrieval/dense.py`

- [ ] **Step 1: Create the file with full implementation**

```python
"""Dense retrieval using BAAI/bge-m3 + FAISS IndexFlatIP."""

import os
import pickle
import sys
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from retrieval.rrf import weighted_rrf

ROOT_DIR    = Path(__file__).resolve().parents[2]
INDEX_DIR   = ROOT_DIR / "indexes"
CORPUS_PATH = INDEX_DIR / "corpus.pkl"
MODEL_PATH  = str(ROOT_DIR / "models" / "bge-m3")

_model:        SentenceTransformer | None = None
_index_court:  faiss.Index         | None = None
_index_law:    faiss.Index         | None = None
_corpus_court: list[dict]          | None = None
_corpus_law:   list[dict]          | None = None


def dense_court_exists() -> bool:
    """Return True if the dense court FAISS index has been built."""
    return (INDEX_DIR / "dense_court" / "index.faiss").exists()


def dense_law_exists() -> bool:
    """Return True if the dense law FAISS index has been built."""
    return (INDEX_DIR / "dense_law" / "index.faiss").exists()


def _load_index(use_court: bool = True, use_law: bool = True) -> None:
    """Lazy-load model, FAISS indexes, and corpus on first call. No-op thereafter."""
    global _model, _index_court, _index_law, _corpus_court, _corpus_law
    if _model is not None:
        return

    print("Loading bge-m3 …", file=sys.stderr)
    try:
        _model = SentenceTransformer(MODEL_PATH, device="mps")
        print("bge-m3 loaded on mps", file=sys.stderr)
    except Exception:
        _model = SentenceTransformer(MODEL_PATH, device="cpu")
        print("bge-m3 loaded on cpu (mps unavailable)", file=sys.stderr)

    if use_court and dense_court_exists():
        print("Loading dense_court/index.faiss …", file=sys.stderr)
        _index_court = faiss.read_index(
            str(INDEX_DIR / "dense_court" / "index.faiss")
        )
        print(f"dense_court ready: {_index_court.ntotal:,} vectors", file=sys.stderr)

    if use_law and dense_law_exists():
        print("Loading dense_law/index.faiss …", file=sys.stderr)
        _index_law = faiss.read_index(
            str(INDEX_DIR / "dense_law" / "index.faiss")
        )
        print(f"dense_law ready: {_index_law.ntotal:,} vectors", file=sys.stderr)

    print("Loading corpus.pkl for dense …", file=sys.stderr)
    with open(CORPUS_PATH, "rb") as f:
        corpus = pickle.load(f)
    _corpus_court = [d for d in corpus if d["source"] == "court"]
    _corpus_law   = [d for d in corpus if d["source"] == "law"]


def retrieve_dense_parts(
    query: str,
    k_court: int = 300,
    k_law: int = 300,
    use_court: bool = True,
    use_law: bool = True,
) -> tuple[list[str], list[str]]:
    """Encode query and search dense indexes.

    Returns (court_citations, law_citations) ranked by cosine similarity.
    Either list is empty if the corresponding index was not loaded.
    """
    _load_index(use_court=use_court, use_law=use_law)

    q_emb = _model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    court_citations: list[str] = []
    if use_court and _index_court is not None:
        k = min(k_court, _index_court.ntotal)
        _, idxs = _index_court.search(q_emb, k)
        court_citations = [_corpus_court[i]["citation"] for i in idxs[0].tolist()]

    law_citations: list[str] = []
    if use_law and _index_law is not None:
        k = min(k_law, _index_law.ntotal)
        _, idxs = _index_law.search(q_emb, k)
        law_citations = [_corpus_law[i]["citation"] for i in idxs[0].tolist()]

    return court_citations, law_citations


def retrieve_dense(
    query: str,
    k: int = 200,
    k_court: int = 300,
    k_law: int = 300,
    weight_court: float = 1.0,
    weight_law: float = 1.2,
    rrf_k: int = 60,
) -> list[str]:
    """Convenience wrapper: dense retrieval with internal RRF fusion.

    Useful for standalone testing. run.py uses retrieve_dense_parts() directly.
    """
    court_citations, law_citations = retrieve_dense_parts(query, k_court, k_law)
    rankings: list[tuple[list[str], float]] = []
    if court_citations:
        rankings.append((court_citations, weight_court))
    if law_citations:
        rankings.append((law_citations, weight_law))
    return weighted_rrf(rankings, rrf_k=rrf_k)[:k]
```

- [ ] **Step 2: Verify import (no FAISS index needed for import check)**

```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python -c "
import sys; sys.path.insert(0,'src')
from retrieval.dense import dense_court_exists, dense_law_exists
print('court index exists:', dense_court_exists())
print('law index exists:',   dense_law_exists())
print('import OK')
"
```
Expected (before any index is built):
```
court index exists: False
law index exists: False
import OK
```

- [ ] **Step 3: Commit**

```bash
git add src/retrieval/dense.py
git commit -m "feat: add dense.py — bge-m3 FAISS retrieval with lazy-load and per-source existence checks"
```

---

## Task 3: Add `retrieve_bm25_parts()` to `src/retrieval/bm25.py`

**Files:**
- Modify: `src/retrieval/bm25.py`

The goal is to expose the raw per-source rankings so `run.py` can include them individually in the 5-way RRF. The existing `retrieve_bm25()` is refactored to call the new function internally — its signature and return type are unchanged.

- [ ] **Step 1: Add `retrieve_bm25_parts()` and refactor `retrieve_bm25()`**

Replace the entire `retrieve_bm25()` function (lines 68–107 in the current file) with the following two functions:

```python
def retrieve_bm25_parts(
    query: str,
    search_text: str | None = None,
    k_court: int = 300,
    k_law: int = 300,
) -> tuple[list[str], list[str], list[str]]:
    """Return (extracted, court_citations, law_citations) without RRF fusion.

    extracted        — citations literally present in the query text
    court_citations  — BM25 court results, ranked by BM25 score
    law_citations    — BM25 law results, ranked by BM25 score
    """
    _load_index()

    extracted = extract_citations_from_query(query)
    text_for_search = search_text if search_text is not None else query
    tokenized_q = tokenize_for_bm25(
        [text_for_search], citations=[extracted], show_progress=False
    )

    court_results, _ = _retriever_court.retrieve(
        tokenized_q, k=min(k_court, len(_corpus_court))
    )
    law_results, _ = _retriever_law.retrieve(
        tokenized_q, k=min(k_law, len(_corpus_law))
    )

    court_citations = [_corpus_court[i]["citation"] for i in court_results[0].tolist()]
    law_citations   = [_corpus_law[i]["citation"]   for i in law_results[0].tolist()]

    return extracted, court_citations, law_citations


def retrieve_bm25(
    query: str,
    search_text: str | None = None,
    k: int = 700,
    k_court: int = 300,
    k_law: int = 300,
    weight_extracted: float = 2.0,
    weight_law: float = 1.2,
    weight_court: float = 1.0,
    rrf_k: int = 60,
) -> list[str]:
    """Return up to k citation strings via dual BM25 + query extraction RRF fusion."""
    extracted, court_citations, law_citations = retrieve_bm25_parts(
        query, search_text, k_court, k_law
    )

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

- [ ] **Step 2: Verify behaviour is unchanged via quick smoke test**

```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python -c "
import sys; sys.path.insert(0,'src')
from retrieval.bm25 import retrieve_bm25, retrieve_bm25_parts

query = 'Art. 8 ZGB Beweislast'
extracted, court, law = retrieve_bm25_parts(query)
print('extracted:', extracted[:3])
print('court[:3]:', court[:3])
print('law[:3]:', law[:3])
fused = retrieve_bm25(query, k=5)
print('fused[:5]:', fused[:5])
print('OK')
"
```
Expected: lists of citation strings printed, no errors.

- [ ] **Step 3: Commit**

```bash
git add src/retrieval/bm25.py
git commit -m "refactor: add retrieve_bm25_parts() to bm25.py; retrieve_bm25() becomes thin wrapper"
```

---

## Task 4: Update `src/query/run.py` — 5-way RRF

**Files:**
- Modify: `src/query/run.py`

- [ ] **Step 1: Replace the import block and add module-level dense flags**

Replace the current imports at the top of `run.py`:

```python
# old
from retrieval.bm25 import _load_index, retrieve_bm25
```

With:

```python
from retrieval.bm25 import _load_index as _load_bm25_index, retrieve_bm25_parts
from retrieval.dense import (
    _load_index as _load_dense_index,
    dense_court_exists,
    dense_law_exists,
    retrieve_dense_parts,
)
from retrieval.rrf import weighted_rrf
```

Then, directly after the `DEFAULT_REWRITE_LOG_DIR` constant line, add:

```python
_USE_DENSE_COURT: bool = dense_court_exists()
_USE_DENSE_LAW:   bool = dense_law_exists()
```

- [ ] **Step 2: Replace `predict_citations()` with the 5-way RRF version**

Replace the entire `predict_citations()` function with:

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
    weight_dense_court: float = 1.0,
    weight_dense_law: float = 1.2,
    rrf_k: int = 60,
    # deprecated aliases kept for backward compatibility
    weight_court: float | None = None,
    weight_law: float | None = None,
) -> list[str]:
    """Query pipeline: optional LLM rewrite + 5-way weighted RRF fusion."""
    if weight_court is not None:
        weight_bm25_court = weight_court
    if weight_law is not None:
        weight_bm25_law = weight_law

    search_text = None
    if use_rewrite:
        rewrite_result = rewrite_query(query)
        search_text = format_search_text(rewrite_result, lang="de")
        if rewrite_log_dir and query_id:
            _log_rewrite(rewrite_log_dir, query_id, query, rewrite_result, search_text)

    extracted, bm25_court, bm25_law = retrieve_bm25_parts(
        query, search_text, k_court, k_law
    )

    rankings: list[tuple[list[str], float]] = [
        (extracted,   weight_extracted),
        (bm25_court,  weight_bm25_court),
        (bm25_law,    weight_bm25_law),
    ]

    if _USE_DENSE_COURT or _USE_DENSE_LAW:
        dense_court, dense_law = retrieve_dense_parts(
            query,
            k_court=k_court,
            k_law=k_law,
            use_court=_USE_DENSE_COURT,
            use_law=_USE_DENSE_LAW,
        )
        if _USE_DENSE_COURT and dense_court:
            rankings.append((dense_court, weight_dense_court))
        if _USE_DENSE_LAW and dense_law:
            rankings.append((dense_law, weight_dense_law))

    return weighted_rrf(rankings, rrf_k=rrf_k)[:k]
```

- [ ] **Step 3: Update `_process_query()` to pass the new weight params**

Replace `_process_query()` with:

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

- [ ] **Step 4: Update `run()` to preload dense index and pass new params**

Replace the `run()` function with:

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
    queries = load_queries(input_path)
    if not queries:
        write_predictions(output_path, [])
        print(f"Wrote 0 predictions → {output_path}", file=sys.stderr)
        return

    _load_bm25_index()
    if _USE_DENSE_COURT or _USE_DENSE_LAW:
        _load_dense_index(use_court=_USE_DENSE_COURT, use_law=_USE_DENSE_LAW)

    process_kwargs = {
        "use_rewrite": use_rewrite,
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

- [ ] **Step 5: Update `main()` CLI args**

Replace the `main()` function with:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description="Run query pipeline and write predictions CSV")
    parser.add_argument("--input",  default=DEFAULT_INPUT,  help="Input query CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output predictions CSV")
    parser.add_argument("--k-court", type=int, default=300)
    parser.add_argument("--k-law",   type=int, default=300)
    parser.add_argument("--k",       type=int, default=200, help="RRF output top-k")
    parser.add_argument("--weight-extracted",   type=float, default=2.0)
    parser.add_argument("--weight-bm25-court",  type=float, default=1.0)
    parser.add_argument("--weight-bm25-law",    type=float, default=1.2)
    parser.add_argument("--weight-dense-court", type=float, default=1.0)
    parser.add_argument("--weight-dense-law",   type=float, default=1.2)
    # deprecated aliases
    parser.add_argument("--weight-court", type=float, default=None,
                        help="Deprecated alias for --weight-bm25-court")
    parser.add_argument("--weight-law",   type=float, default=None,
                        help="Deprecated alias for --weight-bm25-law")
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--no-rewrite", action="store_true", help="Skip LLM query rewrite")
    parser.add_argument(
        "--rewrite-log",
        default=DEFAULT_REWRITE_LOG_DIR,
        help=f"Directory for per-query rewrite JSON logs (default: {DEFAULT_REWRITE_LOG_DIR})",
    )
    parser.add_argument(
        "--no-rewrite-log",
        action="store_true",
        help="Do not write rewrite JSON logs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel worker threads (default: 4; use 1 for sequential)",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")

    weight_bm25_court = args.weight_court if args.weight_court is not None else args.weight_bm25_court
    weight_bm25_law   = args.weight_law   if args.weight_law   is not None else args.weight_bm25_law

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
        rewrite_log_dir=None if args.no_rewrite_log else args.rewrite_log,
        workers=args.workers,
    )
```

- [ ] **Step 6: Smoke test BM25-only path (no dense index needed)**

```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python src/query/run.py \
  --input dataset/val.csv \
  --no-rewrite \
  --output results/predictions_smoke.csv \
  --workers 1
```

Expected stderr:
```
Loading BM25 court index ...
Loading BM25 law index ...
Loading corpus.pkl ...
Ready. Court: ..., law: ...
[1/11] ...
```
No mention of dense (indexes not built yet). Output file `results/predictions_smoke.csv` produced.

- [ ] **Step 7: Commit**

```bash
git add src/query/run.py
git commit -m "feat: 5-way RRF in run.py — BM25 parts + dense parts; new --weight-bm25-* / --weight-dense-* CLI args"
```

---

## Task 5: Manual Validation (Dense Law Index)

These steps are run manually — not automated.

- [ ] **Step 1: Download model (if not already done)**

```bash
conda activate agent
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
```

Verify:
```bash
ls models/bge-m3/config.json
```

- [ ] **Step 2: Build law index first (~36 min)**

```bash
conda activate agent
python src/indexing/build_dense.py --source law
```

Expected final stderr line:
```
[law] index saved → indexes/dense_law/index.faiss  (175,933 vectors)
```

- [ ] **Step 3: Verify dense_law auto-activates in run.py**

```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python src/query/run.py \
  --input dataset/val.csv \
  --no-rewrite \
  --output results/predictions_dense_law.csv \
  --workers 1
```

Expected stderr should now include:
```
Loading bge-m3 …
bge-m3 loaded on mps
Loading dense_law/index.faiss …
dense_law ready: 175,933 vectors
```

- [ ] **Step 4: Build court index (run overnight)**

```bash
conda activate agent
nohup python src/indexing/build_dense.py --source court > logs/build_court.log 2>&1 &
```

(Create `logs/` dir first if needed: `mkdir -p logs`)

Monitor progress:
```bash
tail -f logs/build_court.log
```

- [ ] **Step 5: Verify full 5-way RRF after court index completes**

```bash
/Users/vingo/opt/anaconda3/envs/agent/bin/python src/query/run.py \
  --input dataset/val.csv \
  --no-rewrite \
  --output results/predictions_dense_full.csv \
  --workers 1
```

Expected stderr includes both:
```
dense_court ready: 1,982,724 vectors
dense_law ready: 175,933 vectors
```
