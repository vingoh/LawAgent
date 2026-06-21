import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from retrieval.rrf import weighted_rrf


def test_single_ranking():
    result, scores = weighted_rrf([( ["a", "b", "c"], 1.0)], rrf_k=60)
    assert result == ["a", "b", "c"]
    assert set(scores.keys()) == {"a", "b", "c"}
    assert scores["a"] > scores["b"] > scores["c"]


def test_multi_ranking_accumulates():
    result, scores = weighted_rrf(
        [(["a", "b"], 1.0), (["b", "c"], 1.0)], rrf_k=60
    )
    assert result[0] == "b"
    assert set(result) == {"a", "b", "c"}


def test_weighted_ranking():
    result, scores = weighted_rrf(
        [(["a"], 1.0), (["b"], 3.0)], rrf_k=60
    )
    assert result[0] == "b"


def test_empty_rankings():
    result, scores = weighted_rrf([], rrf_k=60)
    assert result == []
    assert scores == {}


def test_empty_single_list():
    result, scores = weighted_rrf([([], 1.0), (["a"], 1.0)], rrf_k=60)
    assert result == ["a"]
