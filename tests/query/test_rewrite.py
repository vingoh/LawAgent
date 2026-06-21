import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from query.rewrite import format_search_text, parse_rewrite_result, RewriteResult


def _sample_data():
    return {
        "legal_issue": "Untersuchungshaft Verlängerung",
        "expected_codes": ["StPO", "StGB"],
        "search_terms": {
            "de": ["Kollusionsgefahr", "Verhältnismäßigkeit"],
            "fr": ["prolongation détention"],
        },
    }


def test_parse_rewrite_result_valid():
    result = parse_rewrite_result(_sample_data())
    assert isinstance(result, RewriteResult)
    assert result.legal_issue == "Untersuchungshaft Verlängerung"
    assert result.expected_codes == ["StPO", "StGB"]
    assert result.search_terms["de"] == ["Kollusionsgefahr", "Verhältnismäßigkeit"]
    assert result.search_terms["fr"] == ["prolongation détention"]


def test_parse_rewrite_result_missing_fr_defaults_empty():
    data = _sample_data()
    del data["search_terms"]["fr"]
    result = parse_rewrite_result(data)
    assert result.search_terms["fr"] == []


def test_parse_rewrite_result_missing_de_raises():
    data = _sample_data()
    del data["search_terms"]["de"]
    with pytest.raises(ValueError, match="search_terms.de"):
        parse_rewrite_result(data)


def test_parse_rewrite_result_empty_de_raises():
    data = _sample_data()
    data["search_terms"]["de"] = []
    with pytest.raises(ValueError, match="search_terms.de"):
        parse_rewrite_result(data)


def test_format_search_text_default_de():
    result = parse_rewrite_result(_sample_data())
    text = format_search_text(result)
    assert text == (
        "Untersuchungshaft Verlängerung "
        "Kollusionsgefahr Verhältnismäßigkeit"
    )


def test_format_search_text_lang_fr():
    result = parse_rewrite_result(_sample_data())
    text = format_search_text(result, lang="fr")
    assert text == "Untersuchungshaft Verlängerung prolongation détention"


from unittest.mock import patch

from query.rewrite import rewrite_query


def test_rewrite_query_calls_llm_and_parses():
    mock_data = {
        "legal_issue": "Test Issue",
        "expected_codes": ["OR"],
        "search_terms": {"de": ["Term A"], "fr": []},
    }
    with patch("query.rewrite.chat_json", return_value=mock_data) as mock_chat:
        result = rewrite_query("Can a party rescind a contract?")

    assert result.legal_issue == "Test Issue"
    mock_chat.assert_called_once()
    system_prompt, user_prompt = mock_chat.call_args[0]
    assert "JSON" in system_prompt
    assert "Can a party rescind a contract?" in user_prompt


def test_rewrite_query_retries_on_failure():
    mock_data = {
        "legal_issue": "Test Issue",
        "expected_codes": ["OR"],
        "search_terms": {"de": ["Term A"], "fr": []},
    }
    with patch(
        "query.rewrite.chat_json",
        side_effect=[RuntimeError("API error"), mock_data],
    ) as mock_chat:
        result = rewrite_query("test query")

    assert result.legal_issue == "Test Issue"
    assert mock_chat.call_count == 2


def test_rewrite_query_raises_after_both_failures():
    with patch(
        "query.rewrite.chat_json",
        side_effect=RuntimeError("API error"),
    ) as mock_chat:
        with pytest.raises(RuntimeError, match="API error"):
            rewrite_query("test query")

    assert mock_chat.call_count == 2
