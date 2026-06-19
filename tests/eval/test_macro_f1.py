import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from eval.macro_f1 import (
    _query_f1,
    compute_macro_f1,
    parse_citations,
)


def test_parse_citations_splits_and_strips():
    s = "Art. 23 OR; Art. 24 OR ;BGE 136 III 528"
    assert parse_citations(s) == {
        "Art. 23 OR",
        "Art. 24 OR",
        "BGE 136 III 528",
    }


def test_parse_citations_empty():
    assert parse_citations("") == set()
    assert parse_citations("  ;  ") == set()


def test_query_f1_exact_match():
    gold = {"Art. 23 OR", "Art. 24 OR", "BGE 136 III 528 E. 3.4.1"}
    pred = {"Art. 23 OR", "Art. 24 OR", "Art. 31 OR"}
    p, r, f1 = _query_f1(gold, pred)
    assert p == 2 / 3
    assert r == 2 / 3
    assert abs(f1 - 2 / 3) < 1e-9


def test_query_f1_empty_pred():
    p, r, f1 = _query_f1({"Art. 1 OR"}, set())
    assert (p, r, f1) == (0.0, 0.0, 0.0)


def test_compute_macro_f1():
    gold_list = [{"a"}, {"x", "y"}]
    pred_list = [{"a"}, {"x"}]
    assert compute_macro_f1(gold_list, pred_list) == (1.0 + 2 / 3) / 2
