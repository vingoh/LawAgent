"""Dense retrieval using BAAI/bge-m3 + FAISS IndexFlatIP."""

import os
import sys
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from retrieval.rrf import weighted_rrf
from retrieval import corpus as _corpus_mod

ROOT_DIR   = Path(__file__).resolve().parents[2]
INDEX_DIR  = ROOT_DIR / "indexes"
MODEL_PATH = str(ROOT_DIR / "models" / "bge-m3")

_loaded:       bool                       = False
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
    """Lazy-load model, FAISS indexes, and corpus on first call. No-op thereafter.

    NOTE: use_court/use_law arguments are ignored on repeat calls — the first call
    determines what is loaded for the lifetime of the process.
    """
    global _loaded, _model, _index_court, _index_law, _corpus_court, _corpus_law
    if _loaded:
        return

    import torch
    device = (
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()         else
        "cpu"
    )
    print("Loading bge-m3 …", file=sys.stderr)
    _model = SentenceTransformer(MODEL_PATH, device=device)
    print(f"bge-m3 loaded on {device}", file=sys.stderr)

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

    _corpus_court = _corpus_mod.get_corpus_court()
    _corpus_law   = _corpus_mod.get_corpus_law()
    if _index_court is not None and _index_court.ntotal != len(_corpus_court):
        raise RuntimeError(
            f"dense_court: FAISS index has {_index_court.ntotal:,} vectors "
            f"but corpus has {len(_corpus_court):,} docs — rebuild the index."
        )
    if _index_law is not None and _index_law.ntotal != len(_corpus_law):
        raise RuntimeError(
            f"dense_law: FAISS index has {_index_law.ntotal:,} vectors "
            f"but corpus has {len(_corpus_law):,} docs — rebuild the index."
        )
    _loaded = True


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
        k_eff = min(k_court, _index_court.ntotal)
        _, idxs = _index_court.search(q_emb, k_eff)
        court_citations = [_corpus_court[i]["citation"] for i in idxs[0].tolist()]

    law_citations: list[str] = []
    if use_law and _index_law is not None:
        k_eff = min(k_law, _index_law.ntotal)
        _, idxs = _index_law.search(q_emb, k_eff)
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
