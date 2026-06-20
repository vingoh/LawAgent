# Dense Retrieval Design

**Date:** 2026-06-19  
**Status:** Approved  
**Scope:** Add bge-m3 dense vector index alongside existing dual BM25, fuse all signals via 5-way weighted RRF.

---

## Overview

The current pipeline uses two BM25 indexes (court + law) plus query-time citation extraction, fused via weighted RRF. This design adds dense retrieval (BAAI/bge-m3) as two parallel ranking signals (dense_court, dense_law), bringing the outer RRF to 5 paths:

```
query ──► extracted citations        (weight_extracted)
      ──► BM25 court                 (weight_bm25_court)
      ──► BM25 law                   (weight_bm25_law)
      ──► Dense court (bge-m3+FAISS) (weight_dense_court)  ← new
      ──► Dense law   (bge-m3+FAISS) (weight_dense_law)    ← new
                    │
                weighted_rrf
                    │
               top-k citations
```

Dense indexes are optional and per-source independent: if only `dense_law/index.faiss` exists, only the law dense signal is added (4-way RRF); if neither exists, the pipeline runs BM25-only (3-way RRF).

---

## Corpus Facts

| Source | Documents | Text length (median) | Est. tokens (median) |
|--------|-----------|----------------------|----------------------|
| Court  | 1,982,724 | 711 chars            | ~158                 |
| Law    | 175,933   | 349 chars            | ~78                  |
| Total  | 2,158,657 |                      |                      |

- Embeddings float32: court ~8.1 GB, law ~0.7 GB, total ~8.8 GB
- Embeddings float16 (on-disk chunks): court ~4.1 GB, law ~0.36 GB

---

## File Structure

### New files

```
src/indexing/build_dense.py   — offline index builder
src/retrieval/dense.py        — online retrieval module
```

### Modified files

```
src/retrieval/bm25.py         — add retrieve_bm25_parts()
src/query/run.py              — replace 3-way with 5-way RRF; add CLI args
```

### Index directories (parallel to BM25)

```
indexes/
├── bm25_court/               existing
├── bm25_law/                 existing
├── dense_court/              new
│   ├── chunks/               intermediate .npy files (resumable)
│   │   ├── 0000.npy          50 K docs, float16, ~100 MB each
│   │   ├── 0001.npy
│   │   └── …  (~40 chunks)
│   └── index.faiss           IndexFlatIP, float32, ~8.1 GB
└── dense_law/
    ├── chunks/               (~4 chunks)
    └── index.faiss           ~0.7 GB
```

The `chunks/` directories are build artifacts. After `index.faiss` is verified they may be deleted to reclaim disk space, but keeping them allows rebuilding the FAISS index without re-encoding.

---

## `src/indexing/build_dense.py`

### Responsibilities

1. Load `indexes/corpus.pkl`, split into court / law lists.
2. For each source, encode docs in chunks of 50 K with bge-m3 on MPS, save `.npy` chunks.
3. Skip chunks whose `.npy` file already exists (checkpoint resume).
4. After all chunks are written, merge into a FAISS IndexFlatIP and write `index.faiss`.

### Key constants

| Name | Value | Rationale |
|------|-------|-----------|
| `CHUNK_SIZE` | 50,000 | ~100 MB float16 per file; peak memory controllable |
| `ENCODE_BATCH_SIZE` | 16 | Safe batch size for bge-m3 on M1 Pro 16 GB |
| `MAX_LENGTH` | 512 | bge-m3 default; texts beyond this are truncated |
| Storage dtype | float16 | Halves chunk file sizes |
| FAISS dtype | float32 | Required by faiss-cpu |

### Encoding loop (pseudo-code)

```python
model = SentenceTransformer("BAAI/bge-m3", device="mps")

for source, docs in [("court", court_docs), ("law", law_docs)]:
    chunks_dir = INDEX_DIR / f"dense_{source}" / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    for chunk_idx, start in enumerate(range(0, len(docs), CHUNK_SIZE)):
        chunk_path = chunks_dir / f"{chunk_idx:04d}.npy"
        if chunk_path.exists():
            print(f"[{source}] chunk {chunk_idx} exists, skipping")
            continue
        batch_texts = [d["indexed_text"] for d in docs[start:start + CHUNK_SIZE]]
        embs = model.encode(
            batch_texts,
            batch_size=ENCODE_BATCH_SIZE,
            normalize_embeddings=True,   # L2 norm → FlatIP == cosine
            show_progress_bar=True,
        ).astype("float16")
        np.save(chunk_path, embs)
        print(f"[{source}] chunk {chunk_idx} saved ({len(batch_texts):,} docs)")
```

### FAISS merge (memory-friendly)

Load and add one chunk at a time so peak RAM is one chunk (~200 MB float32):

```python
    dim = 1024
    index = faiss.IndexFlatIP(dim)
    for chunk_path in sorted(chunks_dir.glob("*.npy")):
        embs = np.load(chunk_path).astype("float32")
        index.add(embs)
    faiss.write_index(index, str(INDEX_DIR / f"dense_{source}" / "index.faiss"))
```

### Runtime estimates (M1 Pro, MPS)

| Source | Chunks | Encode time | Notes |
|--------|--------|-------------|-------|
| Law    | ~4     | ~36 min     | Run first to validate end-to-end |
| Court  | ~40    | ~6.7 h      | Resume safe; run overnight |

### Entry point

```
conda activate agent
python src/indexing/build_dense.py
python src/indexing/build_dense.py --source law   # law only
python src/indexing/build_dense.py --source court # court only
```

---

## `src/retrieval/dense.py`

### Module-level globals (lazy-loaded)

```python
_model:        SentenceTransformer | None = None
_index_court:  faiss.Index         | None = None
_index_law:    faiss.Index         | None = None
_corpus_court: list[dict]          | None = None
_corpus_law:   list[dict]          | None = None
```

### Public API

#### `dense_court_exists() -> bool` / `dense_law_exists() -> bool`

Each returns `True` if the corresponding `index.faiss` file exists. Called once at process start in `run.py` to set `_USE_DENSE_COURT` and `_USE_DENSE_LAW` independently. This allows building and activating the law index first (for early validation) before the court index finishes.

#### `_load_index() -> None`

Lazy-loads on first call:
1. Load bge-m3 on `device="mps"` (falls back to `cpu` if MPS unavailable).
2. `faiss.read_index(...)` for court and law indexes.
3. Load `corpus.pkl`, split into `_corpus_court` / `_corpus_law`.

Loading the court FlatIP index (~8.1 GB) takes ~20–30 s on first call.

#### `retrieve_dense_parts(query, k_court, k_law) -> tuple[list[str], list[str]]`

```python
def retrieve_dense_parts(
    query: str,
    k_court: int = 300,
    k_law: int = 300,
) -> tuple[list[str], list[str]]:
```

1. `_load_index()`
2. Encode `query` with `normalize_embeddings=True` → shape `(1, 1024)`.
3. `_index_court.search(q_emb, k_court)` → court doc indices.
4. `_index_law.search(q_emb, k_law)` → law doc indices.
5. Return `(court_citations, law_citations)` as ranked `list[str]`.

#### `retrieve_dense(query, k, k_court, k_law, weight_court, weight_law, rrf_k) -> list[str]`

Convenience wrapper: calls `retrieve_dense_parts()` and fuses internally with `weighted_rrf`. Used for standalone testing.

---

## `src/retrieval/bm25.py` — additions

Add a public function `retrieve_bm25_parts()` that exposes the raw per-source rankings before internal fusion:

```python
def retrieve_bm25_parts(
    query: str,
    search_text: str | None = None,
    k_court: int = 300,
    k_law: int = 300,
) -> tuple[list[str], list[str], list[str]]:
    """Return (extracted, court_citations, law_citations) without RRF fusion."""
```

The existing `retrieve_bm25()` becomes a thin wrapper calling `retrieve_bm25_parts()` and then `weighted_rrf()`.

---

## `src/query/run.py` — changes

### `predict_citations()` signature additions

```python
weight_bm25_court:   float = 1.0,   # replaces weight_court
weight_bm25_law:     float = 1.2,   # replaces weight_law
weight_dense_court:  float = 1.0,   # new
weight_dense_law:    float = 1.2,   # new
```

Old `weight_court` / `weight_law` are accepted as deprecated kwargs and map to `weight_bm25_court` / `weight_bm25_law`.

### Fusion logic

```python
_USE_DENSE_COURT: bool = dense_court_exists()
_USE_DENSE_LAW:   bool = dense_law_exists()

def predict_citations(query, *, search_text=None, ...) -> list[str]:
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
            query, k_court, k_law,
            use_court=_USE_DENSE_COURT,
            use_law=_USE_DENSE_LAW,
        )
        if _USE_DENSE_COURT:
            rankings.append((dense_court, weight_dense_court))
        if _USE_DENSE_LAW:
            rankings.append((dense_law,   weight_dense_law))

    return weighted_rrf(rankings, rrf_k)[:k]
```

### New CLI args

| Flag | Default | Notes |
|------|---------|-------|
| `--weight-bm25-court` | 1.0 | Replaces `--weight-court` |
| `--weight-bm25-law` | 1.2 | Replaces `--weight-law` |
| `--weight-dense-court` | 1.0 | Active only when dense index exists |
| `--weight-dense-law` | 1.2 | Active only when dense index exists |

`--weight-court` and `--weight-law` remain as deprecated aliases.

---

## Memory Profile at Query Time

| Component | RAM |
|-----------|-----|
| bge-m3 model | ~2.3 GB |
| FAISS dense_court (FlatIP float32) | ~8.1 GB |
| FAISS dense_law (FlatIP float32) | ~0.7 GB |
| BM25 court index | ~0.5 GB (est.) |
| BM25 law index | ~0.1 GB (est.) |
| corpus.pkl (shared) | ~1.5 GB (est.) |
| **Total** | **~13 GB** |

Running on 16 GB unified memory is feasible but leaves ~3 GB headroom. Close all other heavy applications before running the query pipeline with dense enabled.

---

## Dependencies

New packages required (not currently in `agent` env):

```bash
conda activate agent
pip install sentence-transformers faiss-cpu
```

`sentence-transformers` pulls in `torch` (CPU+MPS build for macOS) automatically.

---

## Model Acquisition

bge-m3 is downloaded manually and stored at `models/bge-m3/` inside the project root. This pins the exact model version and allows offline use.

**Download command:**

```bash
conda activate agent
# If inside China, prepend: HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
```

**Project layout after download:**

```
LawAgent/
└── models/
    └── bge-m3/
        ├── config.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── sentencepiece.bpe.model
        ├── model.safetensors   (~2.3 GB)
        └── …
```

**`.gitignore` addition required:**

```
models/
```

**Reference in code:**

```python
MODEL_PATH = os.path.join(ROOT_DIR, "models/bge-m3")
model = SentenceTransformer(MODEL_PATH, device="mps")
```

`build_dense.py` and `dense.py` both use this `MODEL_PATH` constant, defined once at module level in each file (pointing to the same path).

---

## Error Handling

- `build_dense.py`: atomic chunk writes — save to `<chunk_path>.tmp` first, then `os.rename()` to the final `.npy` path. If encoding crashes mid-chunk, the `.tmp` file is left behind (not a `.npy`), so on restart the chunk is not skipped and is re-encoded cleanly.
- `dense.py`: if `dense_court_exists()` / `dense_law_exists()` return `False` at startup, the corresponding `_USE_DENSE_*` flag is `False` and that source's dense path is never entered — no error is raised.
- `dense.py`: if MPS is unavailable (non-Apple hardware), fall back to `device="cpu"` silently.

---

## Validation Plan

No automated test cases. Index building and evaluation are run manually.

1. Install deps: `pip install sentence-transformers faiss-cpu`
2. Download model: `huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3`
3. Run `python src/indexing/build_dense.py --source law` manually (~36 min); verify `indexes/dense_law/index.faiss` is created.
4. Run `python src/query/run.py --input dataset/val.csv`; confirm dense_law signal activates in stderr output.
5. Run `python src/indexing/build_dense.py --source court` manually (overnight ~6.7 h).
6. Re-run `src/query/run.py` with full 5-way RRF; compare macro-F1 vs BM25-only baseline.
