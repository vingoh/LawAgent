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
import pickle
import sys
from pathlib import Path

import torch

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT_DIR    = Path(__file__).resolve().parents[2]
INDEX_DIR   = ROOT_DIR / "indexes"
CORPUS_PATH = INDEX_DIR / "corpus.pkl"
MODEL_PATH  = str(ROOT_DIR / "models" / "bge-m3")

CHUNK_SIZE        = 50000
ENCODE_BATCH_SIZE = 32
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

    if not docs:
        print(f"[{source}] WARNING: no docs — skipping.", file=sys.stderr)
        return

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
                f"[{source}] chunk {chunk_idx + 1}/{n_chunks} exists, skipping",
                file=sys.stderr,
            )
            continue

        batch = docs[start : start + CHUNK_SIZE]
        texts = [d["indexed_text"] for d in batch]
        print(
            f"[{source}] encoding chunk {chunk_idx + 1}/{n_chunks} ({len(texts):,} docs) …",
            file=sys.stderr,
        )
        embs = model.encode(
            texts,
            batch_size=ENCODE_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float16")

        # Atomic write: .tmp.npy → .npy so partial crashes don't look like done chunks
        # Note: np.save appends .npy if the path doesn't end with .npy, so the
        # temp name must end with .npy to avoid a rename-source-not-found error.
        tmp_path = chunks_dir / f"{chunk_idx:04d}.tmp.npy"
        np.save(str(tmp_path), embs)
        tmp_path.rename(chunk_path)
        print(f"[{source}] chunk {chunk_idx + 1} saved → {chunk_path}", file=sys.stderr)

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

    if index.ntotal != n:
        sys.exit(
            f"[{source}] ERROR: index has {index.ntotal:,} vectors but expected {n:,}. "
            f"Delete chunks/ and re-run."
        )
    faiss.write_index(index, str(index_path))
    print(
        f"[{source}] index saved → {index_path}  ({index.ntotal:,} vectors)",
        file=sys.stderr,
    )


def main() -> None:
    if not CORPUS_PATH.exists():
        sys.exit(f"ERROR: corpus not found: {CORPUS_PATH}")
    model_dir = Path(MODEL_PATH)
    if not model_dir.is_dir():
        sys.exit(
            f"ERROR: model directory not found: {model_dir}\n"
            f"Download with: HF_ENDPOINT=https://hf-mirror.com "
            f"huggingface-cli download BAAI/bge-m3 --local-dir {model_dir}"
        )

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

    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()         else
        "cpu"
    )
    print(f"Using device: {device}", file=sys.stderr)
    print(f"Loading model from {MODEL_PATH} …", file=sys.stderr)
    model = SentenceTransformer(MODEL_PATH, device=device)

    if args.source in ("law", "all"):
        _encode_source(law_docs, "law", model)
    if args.source in ("court", "all"):
        _encode_source(court_docs, "court", model)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
