"""LLM-based query rewriting for legal retrieval."""

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm.client import chat_json

SYSTEM_PROMPT = """You are a Swiss legal query rewriting assistant for legal information retrieval.

Your task is to analyze an English legal question and produce retrieval-oriented German and French legal search terms for Swiss legal sources.

Do NOT answer the legal question.
Do NOT provide legal reasoning.
Do NOT invent article numbers, case citations, or legal citations.
If a citation appears in the input question, do not rewrite it; citation extraction is handled separately.

Output exactly one valid JSON object with this schema:
{
"legal_issue": string,
"expected_codes": string[],
"search_terms": {
"de": string[],
"fr": string[]
}
}

Field requirements:

* "legal_issue": one concise German phrase describing the core legal issue.
* "expected_codes": Swiss code abbreviations that may apply. Use only likely abbreviations such as StPO, StGB, OR, ZGB, ZPO, BGG, BV, ATSG, IVG, AHVG, UVG, AsylG, AIG, SchKG, DSG. Use an empty list if uncertain.
* "search_terms.de": 5-10 German legal search phrases using Swiss legal terminology.
* "search_terms.fr": 0-5 French legal search phrases only if useful.
* Each search phrase should be concise, preferably 2-8 words.
* Include diverse synonyms and doctrinal terms likely to appear in Swiss statutes or court decisions.
* Use natural German/French legal wording.
* Do NOT lowercase, stem, remove stopwords, or tokenize artificially.
* Avoid duplicate or near-duplicate phrases.
* Do not include explanations, Markdown, comments, or text outside the JSON object.
"""


@dataclass
class RewriteResult:
    legal_issue: str
    expected_codes: list[str]
    search_terms: dict[str, list[str]]


def parse_rewrite_result(data: dict) -> RewriteResult:
    legal_issue = data.get("legal_issue", "")
    if not isinstance(legal_issue, str) or not legal_issue.strip():
        raise ValueError("legal_issue must be a non-empty string")

    expected_codes = data.get("expected_codes", [])
    if not isinstance(expected_codes, list):
        raise ValueError("expected_codes must be a list")
    expected_codes = [str(c) for c in expected_codes]

    search_terms = data.get("search_terms")
    if not isinstance(search_terms, dict):
        raise ValueError("search_terms must be a dict")
    if "de" not in search_terms:
        raise ValueError("search_terms.de is required")
    de_terms = search_terms["de"]
    if not isinstance(de_terms, list) or not de_terms:
        raise ValueError("search_terms.de must be a non-empty list")
    de_terms = [str(t) for t in de_terms]

    fr_terms = search_terms.get("fr", [])
    if fr_terms is None:
        fr_terms = []
    if not isinstance(fr_terms, list):
        raise ValueError("search_terms.fr must be a list")
    fr_terms = [str(t) for t in fr_terms]

    return RewriteResult(
        legal_issue=legal_issue.strip(),
        expected_codes=expected_codes,
        search_terms={"de": de_terms, "fr": fr_terms},
    )


def format_search_text(result: RewriteResult, lang: str = "de") -> str:
    """Join legal_issue + search_terms[lang] into a single search string."""
    parts = list(result.search_terms.get(lang, []))
    if result.legal_issue:
        parts = [result.legal_issue] + parts
    return " ".join(parts)


def rewrite_query(query: str) -> RewriteResult:
    data = chat_json(SYSTEM_PROMPT, query)
    if not isinstance(data, dict):
        raise ValueError(f"LLM response must be a JSON object, got: {type(data)}")
    try:
        return parse_rewrite_result(data)
    except ValueError as exc:
        raise ValueError(
            f"Invalid rewrite schema: {exc}\nRaw JSON: {json.dumps(data, ensure_ascii=False)}"
        ) from exc
