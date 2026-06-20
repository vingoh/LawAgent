import os
import sys
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import retrieval.corpus as corpus_mod


def _make_corpus(tmp_path: str) -> str:
    """Write a minimal corpus.pkl for tests."""
    docs = [
        {"citation": "Art. 1 OR",  "source": "law",   "indexed_text": "text law 1"},
        {"citation": "Art. 2 OR",  "source": "law",   "indexed_text": "text law 2"},
        {"citation": "BGE 1 I 1",  "source": "court", "indexed_text": "text court 1"},
    ]
    path = os.path.join(tmp_path, "corpus.pkl")
    with open(path, "wb") as f:
        pickle.dump(docs, f)
    return path


def _reset_corpus():
    """Reset module-level state between tests."""
    corpus_mod._corpus_court = None
    corpus_mod._corpus_law = None
    corpus_mod._corpus_texts = None


def test_load_corpus_splits_by_source(tmp_path, monkeypatch):
    _reset_corpus()
    corpus_path = _make_corpus(str(tmp_path))
    monkeypatch.setattr(corpus_mod, "CORPUS_PATH", corpus_path)

    corpus_mod.load_corpus()

    assert len(corpus_mod.get_corpus_court()) == 1
    assert len(corpus_mod.get_corpus_law()) == 2
    assert corpus_mod.get_corpus_court()[0]["citation"] == "BGE 1 I 1"


def test_load_corpus_is_idempotent(tmp_path, monkeypatch):
    _reset_corpus()
    corpus_path = _make_corpus(str(tmp_path))
    monkeypatch.setattr(corpus_mod, "CORPUS_PATH", corpus_path)

    corpus_mod.load_corpus()
    court_id = id(corpus_mod._corpus_court)
    corpus_mod.load_corpus()  # second call
    assert id(corpus_mod._corpus_court) == court_id  # same object, not reloaded


def test_get_corpus_texts(tmp_path, monkeypatch):
    _reset_corpus()
    corpus_path = _make_corpus(str(tmp_path))
    monkeypatch.setattr(corpus_mod, "CORPUS_PATH", corpus_path)

    corpus_mod.load_corpus()
    texts = corpus_mod.get_corpus_texts()

    assert texts["Art. 1 OR"] == "text law 1"
    assert texts["Art. 2 OR"] == "text law 2"
    assert texts["BGE 1 I 1"] == "text court 1"


def test_get_corpus_texts_is_cached(tmp_path, monkeypatch):
    _reset_corpus()
    corpus_path = _make_corpus(str(tmp_path))
    monkeypatch.setattr(corpus_mod, "CORPUS_PATH", corpus_path)

    corpus_mod.load_corpus()
    t1 = corpus_mod.get_corpus_texts()
    t2 = corpus_mod.get_corpus_texts()
    assert t1 is t2  # same dict object


def test_load_corpus_missing_file(monkeypatch):
    _reset_corpus()
    monkeypatch.setattr(corpus_mod, "CORPUS_PATH", "/nonexistent/corpus.pkl")
    try:
        corpus_mod.load_corpus()
        assert False, "Should have raised"
    except FileNotFoundError:
        pass
