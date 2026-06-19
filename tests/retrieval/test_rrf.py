import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from retrieval.rrf import weighted_rrf


def test_single_ranking():
    rankings = [(["a", "b", "c"], 1.0)]
    assert weighted_rrf(rankings, rrf_k=60) == ["a", "b", "c"]


def test_multi_ranking_accumulates():
    rankings = [
        (["a", "b"], 1.0),
        (["b", "c"], 1.0),
    ]
    result = weighted_rrf(rankings, rrf_k=60)
    assert result[0] == "b"
    assert set(result) == {"a", "b", "c"}


def test_weighted_ranking():
    rankings = [
        (["a"], 1.0),
        (["b"], 3.0),
    ]
    result = weighted_rrf(rankings, rrf_k=60)
    assert result[0] == "b"


def test_empty_rankings():
    assert weighted_rrf([], rrf_k=60) == []


def test_empty_single_list():
    assert weighted_rrf([([], 1.0), (["a"], 1.0)], rrf_k=60) == ["a"]
