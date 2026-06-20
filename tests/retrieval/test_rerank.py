import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import retrieval.rerank as rerank_mod


CORPUS_TEXTS = {
    "Art. 1 OR": "Erster Artikel des OR",
    "Art. 2 OR": "Zweiter Artikel des OR",
    "BGE 1 I 1": "Bundesgericht Entscheid",
    "Art. 3 OR": "Dritter Artikel",
}


def _reset_reranker():
    """Reset module-level model state between tests."""
    rerank_mod._reranker_model = None
    rerank_mod._reranker_tok = None
    rerank_mod._reranker_device = None


def test_rerank_reorders_by_score():
    _reset_reranker()
    # Scores: Art.2 highest (0.9), Art.1 middle (0.5), BGE lowest (0.1)
    with patch.object(rerank_mod, "_compute_scores", return_value=[0.5, 0.9, 0.1]):
        candidates = ["Art. 1 OR", "Art. 2 OR", "BGE 1 I 1"]
        result = rerank_mod.rerank("test query", candidates, CORPUS_TEXTS, top_k=3, batch_size=32)

    assert result[0] == "Art. 2 OR"
    assert result[1] == "Art. 1 OR"
    assert result[2] == "BGE 1 I 1"


def test_rerank_appends_tail_beyond_top_k():
    _reset_reranker()
    # Only top_k=2 candidates are sent to reranker; Art.3 is tail
    with patch.object(rerank_mod, "_compute_scores", return_value=[0.8, 0.3]):
        candidates = ["Art. 1 OR", "Art. 2 OR", "Art. 3 OR"]
        result = rerank_mod.rerank("test query", candidates, CORPUS_TEXTS, top_k=2, batch_size=32)

    assert result[:2] == ["Art. 1 OR", "Art. 2 OR"]
    assert result[2] == "Art. 3 OR"


def test_rerank_missing_text_appended_last():
    _reset_reranker()
    sparse_corpus = {"Art. 1 OR": "some text"}
    # Only Art.1 has text, so only 1 score is returned
    with patch.object(rerank_mod, "_compute_scores", return_value=[0.7]):
        candidates = ["Art. 1 OR", "Art. 2 OR"]
        result = rerank_mod.rerank("test query", candidates, sparse_corpus, top_k=2, batch_size=32)

    assert result[0] == "Art. 1 OR"
    assert result[1] == "Art. 2 OR"


def test_rerank_empty_candidates():
    _reset_reranker()
    with patch.object(rerank_mod, "_compute_scores") as mock_score:
        result = rerank_mod.rerank("query", [], CORPUS_TEXTS, top_k=100)
        mock_score.assert_not_called()
    assert result == []


def test_reranker_exists_false_when_no_model(monkeypatch, tmp_path):
    monkeypatch.setattr(rerank_mod, "RERANKER_PATH", str(tmp_path / "nonexistent"))
    assert rerank_mod.reranker_exists() is False


def test_reranker_exists_true_when_model_present(monkeypatch, tmp_path):
    model_dir = tmp_path / "bge-reranker-v2-m3"
    model_dir.mkdir()
    monkeypatch.setattr(rerank_mod, "RERANKER_PATH", str(model_dir))
    assert rerank_mod.reranker_exists() is True
