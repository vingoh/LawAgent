import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import retrieval.selector as sel
from retrieval.selector import (
    Candidate,
    adaptive_count,
    assemble,
    build_candidates,
    cheap_expand,
    fuse_scores,
    llm_verify,
    run_selector,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_candidate(
    citation: str,
    rerank_score: float = 0.5,
    rrf_score: float = 0.1,
    source_set=None,
    direct_regex_hit: bool = False,
    expected_code_match: bool = False,
    expansion_type=None,
    relevance=None,
) -> Candidate:
    return Candidate(
        citation=citation,
        rerank_score=rerank_score,
        rrf_score=rrf_score,
        source_set=frozenset(source_set or ["bm25_court"]),
        direct_regex_hit=direct_regex_hit,
        expected_code_match=expected_code_match,
        expansion_type=expansion_type,
        relevance=relevance,
    )


CORPUS_TEXTS = {
    "BGE 145 IV 154 E. 1.1": "Bundesgericht entschied über Betrug",
    "Art. 146 StGB": "Wer in der Absicht, sich zu bereichern",
    "Art. 221 StPO": "Die Untersuchungshaft",
}

# ── build_candidates ──────────────────────────────────────────────────────────

class FakeQueryInfo:
    expected_codes = ["StGB", "OR"]
    expected_articles = []
    legal_issue = "Betrug"
    search_terms = {"de": [], "fr": []}


def test_build_candidates_fills_source_set():
    scored = [("Art. 146 StGB", 0.9), ("BGE 145 IV 154 E. 1.1", 0.7)]
    rrf_scores = {"Art. 146 StGB": 0.05, "BGE 145 IV 154 E. 1.1": 0.04}
    source_rankings = {
        "bm25_law": ["Art. 146 StGB", "BGE 145 IV 154 E. 1.1"],
        "dense_court": ["BGE 145 IV 154 E. 1.1"],
    }
    candidates = build_candidates(scored, rrf_scores, source_rankings, FakeQueryInfo())
    art = next(c for c in candidates if "StGB" in c.citation and "Art" in c.citation)
    bge = next(c for c in candidates if "BGE" in c.citation)
    assert "bm25_law" in art.source_set
    assert "dense_court" not in art.source_set
    assert "dense_court" in bge.source_set
    assert "bm25_law" in bge.source_set


def test_build_candidates_direct_regex_hit():
    scored = [("Art. 146 StGB", 0.8)]
    rrf_scores = {"Art. 146 StGB": 0.05}
    source_rankings = {
        "extracted": ["Art. 146 StGB"],
        "bm25_law": ["Art. 146 StGB"],
    }
    candidates = build_candidates(scored, rrf_scores, source_rankings, None)
    assert candidates[0].direct_regex_hit is True


def test_build_candidates_llm_not_in_extracted_no_direct_hit():
    """LLM-injected articles not in extracted channel must not get direct_regex_hit."""
    scored = [("Art. 273 ZGB", 0.0), ("Art. 146 StGB", 0.8)]
    rrf_scores = {"Art. 273 ZGB": 0.02, "Art. 146 StGB": 0.05}
    source_rankings = {
        "extracted": [],
        "bm25_law": ["Art. 273 ZGB", "Art. 146 StGB"],
    }
    candidates = build_candidates(scored, rrf_scores, source_rankings, None)
    art273 = next(c for c in candidates if c.citation == "Art. 273 ZGB")
    assert art273.direct_regex_hit is False


def test_build_candidates_expected_code_match():
    scored = [("Art. 146 StGB", 0.8), ("BGE 145 IV 154 E. 1.1", 0.6)]
    rrf_scores = {"Art. 146 StGB": 0.05, "BGE 145 IV 154 E. 1.1": 0.04}
    source_rankings = {"bm25_law": ["Art. 146 StGB", "BGE 145 IV 154 E. 1.1"]}
    candidates = build_candidates(scored, rrf_scores, source_rankings, FakeQueryInfo())
    art = next(c for c in candidates if "Art. 146" in c.citation)
    bge = next(c for c in candidates if "BGE" in c.citation)
    assert art.expected_code_match is True    # StGB in expected_codes
    assert bge.expected_code_match is False   # BGE has no law code


def test_build_candidates_no_query_info():
    scored = [("Art. 146 StGB", 0.8)]
    rrf_scores = {"Art. 146 StGB": 0.05}
    candidates = build_candidates(scored, rrf_scores, {"bm25_law": ["Art. 146 StGB"]}, None)
    assert candidates[0].expected_code_match is False
    assert candidates[0].direct_regex_hit is False

# ── cheap_expand ──────────────────────────────────────────────────────────────

def test_cheap_expand_marks_same_parent():
    """High-score BGE seed → same-parent candidate in pool gets marked."""
    c_seed = _make_candidate("BGE 145 IV 154 E. 1.1", rerank_score=0.8)
    c_sibling = _make_candidate("BGE 145 IV 154 E. 2.3", rerank_score=0.3)
    c_other = _make_candidate("Art. 146 StGB", rerank_score=0.6)
    candidates = [c_seed, c_sibling, c_other]
    result = cheap_expand(candidates, {}, {}, None)
    sibling = next(c for c in result if c.citation == "BGE 145 IV 154 E. 2.3")
    assert sibling.expansion_type == "same_parent"
    # seed itself should NOT be marked
    seed = next(c for c in result if c.citation == "BGE 145 IV 154 E. 1.1")
    assert seed.expansion_type is None


def test_cheap_expand_marks_same_code():
    """High-score StPO seed → other StPO candidate in pool gets marked."""
    c_seed = _make_candidate("Art. 221 StPO", rerank_score=0.75)
    c_same_code = _make_candidate("Art. 237 StPO", rerank_score=0.2)
    candidates = [c_seed, c_same_code]
    result = cheap_expand(candidates, {}, {}, None)
    marked = next(c for c in result if c.citation == "Art. 237 StPO")
    assert marked.expansion_type == "same_code"


def test_cheap_expand_adds_rescue_candidates():
    """Citations in source_rankings[:rescue_top_k] but not in pool get added."""
    c_existing = _make_candidate("Art. 146 StGB", rerank_score=0.8)
    candidates = [c_existing]
    source_rankings = {
        "bm25_law": ["Art. 146 StGB", "Art. 147 StGB"],  # Art. 147 is NOT in pool
    }
    rrf_scores = {"Art. 146 StGB": 0.05, "Art. 147 StGB": 0.03}
    result = cheap_expand(candidates, source_rankings, rrf_scores, None, rescue_top_k=5)
    citations = [c.citation for c in result]
    assert "Art. 147 StGB" in citations
    rescue = next(c for c in result if c.citation == "Art. 147 StGB")
    assert rescue.expansion_type == "bm25_law_rescue"
    assert rescue.rerank_score == 0.0
    assert rescue.rrf_score == pytest.approx(0.03)


def test_cheap_expand_does_not_duplicate_rescue():
    """Candidate already in pool is not added again as rescue."""
    c = _make_candidate("Art. 146 StGB", rerank_score=0.8)
    source_rankings = {"bm25_law": ["Art. 146 StGB"]}
    result = cheap_expand([c], source_rankings, {}, None, rescue_top_k=5)
    assert sum(1 for x in result if x.citation == "Art. 146 StGB") == 1

# ── fuse_scores ───────────────────────────────────────────────────────────────

def test_fuse_scores_direct_regex_hit_boost():
    c = _make_candidate("Art. 146 StGB", rerank_score=0.5, direct_regex_hit=True)
    fuse_scores([c])
    assert c.final_score == pytest.approx(0.5 + 0.25, abs=1e-6)


def test_fuse_scores_llm_relevance_boost():
    c = _make_candidate("Art. 146 StGB", rerank_score=0.5, relevance=3)
    fuse_scores([c])
    assert c.final_score == pytest.approx(0.5 + 0.35, abs=1e-6)


def test_fuse_scores_rescue_penalty():
    c = _make_candidate("Art. 147 StGB", rerank_score=0.0, expansion_type="bm25_law_rescue")
    fuse_scores([c])
    assert c.final_score == pytest.approx(0.0 - 0.05, abs=1e-6)


def test_fuse_scores_direct_hit_plus_rescue_penalty():
    """direct_regex_hit AND expansion_type is not None → -0.05 extra."""
    c = _make_candidate(
        "Art. 146 StGB", rerank_score=0.0,
        direct_regex_hit=True, expansion_type="extracted_rescue"
    )
    fuse_scores([c])
    # +0.25 (direct) -0.05 (rescue) -0.05 (direct+expansion) = +0.15
    assert c.final_score == pytest.approx(0.0 + 0.25 - 0.05 - 0.05, abs=1e-6)


def test_fuse_scores_sorted_descending():
    candidates = [
        _make_candidate("A", rerank_score=0.3),
        _make_candidate("B", rerank_score=0.8),
        _make_candidate("C", rerank_score=0.1),
    ]
    result = fuse_scores(candidates)
    scores = [c.final_score for c in result]
    assert scores == sorted(scores, reverse=True)

# ── adaptive_count ────────────────────────────────────────────────────────────

def test_adaptive_count_respects_min():
    candidates = [_make_candidate(f"C{i}", rerank_score=0.9 - i * 0.1) for i in range(2)]
    for c in candidates:
        c.final_score = c.rerank_score
    assert adaptive_count(candidates, n_llm=1, min_val=3, max_val=40) >= 3


def test_adaptive_count_respects_max():
    candidates = [_make_candidate(f"C{i}", rerank_score=0.9) for i in range(100)]
    for c in candidates:
        c.final_score = 0.9
    with patch.object(sel, "_elbow", return_value=100):
        assert adaptive_count(candidates, n_llm=100, min_val=3, max_val=40) <= 40


def test_adaptive_count_formula():
    """n_final = clamp(round(0.2*n_llm + 0.4*n_elbow + 0.4*n_rel23), 3, 40)."""
    candidates = [_make_candidate(f"C{i}") for i in range(5)]
    for c in candidates:
        c.final_score = 0.5
    with patch.object(sel, "_elbow", return_value=8):
        n = adaptive_count(candidates, n_llm=7, min_val=3, max_val=40)
    # n_rel23=0 (all relevance=None)
    expected = round(0.2 * 7 + 0.4 * 8 + 0.4 * 0)  # round(4.6) = 5
    assert n == max(3, min(40, expected))


def test_adaptive_count_includes_rel23():
    """relevance 2/3 candidates contribute to n_rel23."""
    candidates = [
        _make_candidate(f"C{i}", relevance=3 if i < 5 else 0) for i in range(10)
    ]
    for c in candidates:
        c.final_score = 0.5
    with patch.object(sel, "_elbow", return_value=8):
        n = adaptive_count(candidates, n_llm=7, min_val=3, max_val=40)
    # n_rel23=5; expected = round(0.2*7 + 0.4*8 + 0.4*5) = round(6.6) = 7
    assert n == 7


def test_adaptive_count_rel2_counts():
    """relevance=2 counts; relevance=1 does not."""
    candidates = [
        _make_candidate(f"C{i}", relevance=2 if i < 3 else 1) for i in range(10)
    ]
    for c in candidates:
        c.final_score = 0.5
    with patch.object(sel, "_elbow", return_value=5):
        n = adaptive_count(candidates, n_llm=10, min_val=3, max_val=40)
    # n_rel23=3; expected = round(0.2*10 + 0.4*5 + 0.4*3) = round(5.2) = 5
    assert n == 5

# ── assemble ──────────────────────────────────────────────────────────────────

def test_assemble_top_n():
    candidates = [_make_candidate(f"C{i}") for i in range(10)]
    for i, c in enumerate(candidates):
        c.final_score = 1.0 - i * 0.1
    result = assemble(candidates, n_final=3)
    assert result == ["C0", "C1", "C2"]


def test_assemble_parent_constraint():
    """Same BGE parent: at most max_per_parent=2 citations kept."""
    cs = [
        _make_candidate("BGE 145 IV 154 E. 1.1", rerank_score=0.9),
        _make_candidate("BGE 145 IV 154 E. 2.0", rerank_score=0.8),
        _make_candidate("BGE 145 IV 154 E. 3.0", rerank_score=0.7),
        _make_candidate("Art. 146 StGB", rerank_score=0.6),
    ]
    for i, c in enumerate(cs):
        c.final_score = 1.0 - i * 0.1
    result = assemble(cs, n_final=4, max_per_parent=2)
    bge_count = sum(1 for r in result if "BGE 145 IV 154" in r)
    assert bge_count == 2
    assert "Art. 146 StGB" in result


def test_assemble_direct_hit_overflow():
    """direct_regex_hit candidate beyond n_final is appended."""
    cs = [
        _make_candidate("Art. 146 StGB", rerank_score=0.9),
        _make_candidate("Art. 221 StPO", rerank_score=0.8),
        _make_candidate("Art. 237 StPO", rerank_score=0.7, direct_regex_hit=True),
    ]
    for i, c in enumerate(cs):
        c.final_score = 1.0 - i * 0.1
    result = assemble(cs, n_final=2)
    assert result[:2] == ["Art. 146 StGB", "Art. 221 StPO"]
    assert "Art. 237 StPO" in result   # forced in via direct_regex_hit


def test_assemble_drops_non_corpus():
    """Non-corpus citations are excluded even with direct_regex_hit overflow."""
    cs = [
        _make_candidate("Art. 146 StGB", rerank_score=0.9),
        _make_candidate("Art. 273 ZGB", rerank_score=0.8, direct_regex_hit=True),
    ]
    for i, c in enumerate(cs):
        c.final_score = 1.0 - i * 0.1
    result = assemble(cs, n_final=1, valid_citations=set(CORPUS_TEXTS))
    assert result == ["Art. 146 StGB"]
    assert "Art. 273 ZGB" not in result


def test_assemble_rescue_floor_removes_low_quality():
    """Rescue candidate with rerank_score < floor is excluded."""
    cs = [
        _make_candidate("Art. 146 StGB", rerank_score=0.9),
        _make_candidate("Art. 999 ZGB", rerank_score=0.05, expansion_type="bm25_court_rescue"),
    ]
    for i, c in enumerate(cs):
        c.final_score = 1.0 - i * 0.1
    result = assemble(cs, n_final=5, rescue_score_floor=0.1)
    assert "Art. 999 ZGB" not in result


def test_assemble_dedup():
    """Duplicate citations appear only once."""
    c1 = _make_candidate("Art. 146 StGB", rerank_score=0.9)
    c2 = _make_candidate("Art. 146 StGB", rerank_score=0.5)
    c1.final_score = 0.9
    c2.final_score = 0.5
    result = assemble([c1, c2], n_final=5)
    assert result.count("Art. 146 StGB") == 1

# ── llm_verify ────────────────────────────────────────────────────────────────

def test_llm_verify_fills_relevance():
    c = _make_candidate("Art. 146 StGB", rerank_score=0.9)
    fake_response = {
        "candidate_scores": [{"id": "C000", "relevance": 3, "reason": "direct"}],
        "estimated_answer_count": 7,
    }
    # chat_json is imported at module level in selector.py → patch via selector module
    with patch("retrieval.selector.chat_json", return_value=fake_response):
        result_candidates, n_llm = llm_verify(
            [c], "fraud query", FakeQueryInfo(), CORPUS_TEXTS
        )
    assert result_candidates[0].relevance == 3
    assert result_candidates[0].llm_reason == "direct"
    assert n_llm == 7


def test_llm_verify_fallback_on_failure():
    """Two failures → relevance stays None, n_LLM = 10."""
    c = _make_candidate("Art. 146 StGB", rerank_score=0.9)
    with patch("retrieval.selector.chat_json", side_effect=RuntimeError("API error")):
        result_candidates, n_llm = llm_verify(
            [c], "query", FakeQueryInfo(), CORPUS_TEXTS
        )
    assert result_candidates[0].relevance is None
    assert n_llm == 10

# ── run_selector (integration) ────────────────────────────────────────────────

def test_run_selector_end_to_end():
    scored = [("Art. 146 StGB", 0.9), ("BGE 145 IV 154 E. 1.1", 0.7)]
    rrf_scores = {"Art. 146 StGB": 0.05, "BGE 145 IV 154 E. 1.1": 0.04}
    source_rankings = {"bm25_law": ["Art. 146 StGB", "BGE 145 IV 154 E. 1.1"]}
    fake_response = {
        "candidate_scores": [
            {"id": "C000", "relevance": 3, "reason": "direct"},
            {"id": "C001", "relevance": 1, "reason": "weak"},
        ],
        "estimated_answer_count": 2,
    }
    with patch("retrieval.selector.chat_json", return_value=fake_response):
        result = run_selector(
            "fraud query", FakeQueryInfo(), scored, rrf_scores,
            source_rankings, CORPUS_TEXTS,
        )
    assert isinstance(result, list)
    assert len(result) >= 1
    assert all(isinstance(c, str) for c in result)
    # Art. 146 StGB has relevance=3 (highest) — must appear in result
    assert "Art. 146 StGB" in result


def test_run_selector_filters_non_corpus_scored():
    scored = [("Art. 146 StGB", 0.9), ("Art. 999 ZGB", 0.8)]
    rrf_scores = {"Art. 146 StGB": 0.05, "Art. 999 ZGB": 0.04}
    source_rankings = {"bm25_law": ["Art. 146 StGB", "Art. 999 ZGB"]}
    fake_response = {
        "candidate_scores": [{"id": "C000", "relevance": 3, "reason": "direct"}],
        "estimated_answer_count": 1,
    }
    with patch("retrieval.selector.chat_json", return_value=fake_response):
        result = run_selector(
            "fraud query", FakeQueryInfo(), scored, rrf_scores,
            source_rankings, CORPUS_TEXTS,
        )
    assert "Art. 999 ZGB" not in result
    assert "Art. 146 StGB" in result


def test_apply_verifier_response_rejects_bool():
    from retrieval.selector import _apply_verifier_response

    class _FakeCand:
        relevance = None
        llm_reason = None

    idc = {"C000": _FakeCand()}
    n = _apply_verifier_response({"candidate_scores": [], "estimated_answer_count": True}, idc)
    assert n == 10
