"""Shared BM25 tokenization: lowercase, German stopwords, Snowball stem, citation tokens."""

import re
from typing import Union

import bm25s
import Stemmer

_GERMAN_STEMMER = Stemmer.Stemmer("german")


def citation_to_token(citation: str) -> str:
    """Normalize a citation to a single BM25 token (lowercase, non-word chars → underscores)."""
    s = citation.lower().strip()
    s = re.sub(r"[^\w]+", "_", s).strip("_")
    return s


def _normalize_citations_per_doc(
    texts: list[str],
    citations: Union[list[str], list[list[str]]],
) -> list[list[str]]:
    if not citations:
        return [[] for _ in texts]
    if len(citations) == len(texts):
        if isinstance(citations[0], list):
            return citations  # type: ignore[return-value]
        return [[c] for c in citations]  # type: ignore[misc]
    if len(texts) == 1 and isinstance(citations[0], str):
        return [list(citations)]  # type: ignore[arg-type]
    raise ValueError(
        f"citations length ({len(citations)}) must match texts ({len(texts)}) "
        "or be a flat citation list for a single query"
    )


def _enrich_texts_with_citations(
    texts: list[str],
    citations_per_doc: list[list[str]],
) -> list[str]:
    """Append each full citation as one atomic token at the end of the text."""
    enriched: list[str] = []
    for text, citations in zip(texts, citations_per_doc):
        parts = [text]
        for citation in citations:
            tok = citation_to_token(citation)
            if tok:
                parts.append(tok)
        enriched.append(" ".join(parts))
    return enriched


def tokenize_for_bm25(
    texts: Union[str, list[str]],
    *,
    citations: Union[list[str], list[list[str]], None] = None,
    show_progress: bool = True,
) -> bm25s.tokenization.Tokenized:
    if isinstance(texts, str):
        texts = [texts]

    if citations is not None:
        citations_per_doc = _normalize_citations_per_doc(texts, citations)
        texts = _enrich_texts_with_citations(texts, citations_per_doc)

    return bm25s.tokenize(
        texts,
        lower=True,
        stopwords="german",
        stemmer=_GERMAN_STEMMER,
        show_progress=show_progress,
    )
