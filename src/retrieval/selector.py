"""Post-reranker selector pipeline.

Pipeline order (all functions are in this module):
    build_candidates() → cheap_expand() → llm_verify() → fuse_scores()
    → adaptive_count() → assemble()

Entry point: run_selector()
"""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm.client import chat_json

_BGE_PARENT_RE = re.compile(
    r"^(BGE\s+\d+\s+(?:I{1,3}[ab]?|IV[ab]?|VI{0,3}|IX|X[IVX]*)\s+\d+)(?:\s+E\..*)?$",
    re.IGNORECASE,
)
_LAW_CODE_RE = re.compile(
    r"\b(StPO|StGB|OR|ZGB|ZPO|BGG|BV|ATSG|IVG|AHVG|UVG|AsylG|AIG|SchKG|DSG)\b"
)


@dataclass
class Candidate:
    # ── Core fields ───────────────────────────────────────────────────────
    citation: str
    rerank_score: float          # bge-reranker sigmoid [0, 1]; 0.0 for rescue candidates
    rrf_score: float             # weighted_rrf fused score

    # ── Metadata (filled by build_candidates / cheap_expand) ──────────────
    source_set: frozenset        # retrieval paths that found this citation
    direct_regex_hit: bool       # True if citation was in the extracted channel
    expected_code_match: bool    # True if citation's law code is in expected_codes
    expansion_type: str | None   # None / "same_parent" / "same_code" / "<src>_rescue"

    # ── LLM verifier output (None when verifier is skipped) ───────────────
    relevance: int | None = None       # 0-3
    llm_reason: str | None = None

    # ── Score fusion output ───────────────────────────────────────────────
    final_score: float = 0.0


def _extract_bge_parent(citation: str) -> str | None:
    """Return the BGE parent citation (without Erwägung), or None."""
    m = _BGE_PARENT_RE.match(citation)
    return m.group(1) if m else None


def _extract_law_code(citation: str) -> str | None:
    """Return the Swiss law code abbreviation from a citation, or None."""
    m = _LAW_CODE_RE.search(citation)
    return m.group(1) if m else None


def _is_seed(c: Candidate, threshold: float = 0.5) -> bool:
    """Return True if candidate qualifies as a seed for cheap expansion.

    A seed is either confidently reranked (score >= threshold) or a direct
    regex hit from the extracted channel, regardless of reranker score.
    """
    return c.rerank_score >= threshold or c.direct_regex_hit


def build_candidates(
    scored: list[tuple[str, float]],
    rrf_scores: dict[str, float],
    source_rankings: dict[str, list[str]],
    query_info,                              # RewriteResult or None
) -> list[Candidate]:
    """Build Candidate objects from reranker-scored results.

    Args:
        scored:          Output of rerank_with_scores(): [(citation, score), ...]
        rrf_scores:      Output of weighted_rrf(): citation -> rrf_score dict
        source_rankings: Per-retrieval-path citation lists
                         {"extracted": [...], "bm25_court": [...], ...}
        query_info:      RewriteResult (for expected_codes); may be None
    """
    source_sets = {src: set(ranking) for src, ranking in source_rankings.items()}
    extracted_set: set[str] = source_sets.get("extracted", set())
    expected_codes: set[str] = (
        set(query_info.expected_codes) if query_info is not None else set()
    )

    candidates: list[Candidate] = []
    for citation, rerank_score in scored:
        sources = frozenset(src for src, s in source_sets.items() if citation in s)
        direct = citation in extracted_set
        code = _extract_law_code(citation)
        code_match = code is not None and code in expected_codes
        candidates.append(
            Candidate(
                citation=citation,
                rerank_score=rerank_score,
                rrf_score=rrf_scores.get(citation, 0.0),
                source_set=sources,
                direct_regex_hit=direct,
                expected_code_match=code_match,
                expansion_type=None,
            )
        )
    return candidates


def cheap_expand(
    candidates: list[Candidate],
    source_rankings: dict[str, list[str]],
    rrf_scores: dict[str, float],
    query_info,                              # RewriteResult or None
    *,
    rescue_top_k: int = 10,
    seed_score_threshold: float = 0.5,
) -> list[Candidate]:
    """Mark same_parent / same_code candidates and add source diversity rescue.

    same_parent / same_code: flags are set on candidates already present in the
    scored pool (candidates that share a parent/code but weren't included in the
    reranker's top-k are absent from the pool entirely and won't be added here).

    source diversity rescue: candidates from source_rankings[:rescue_top_k] that
    are NOT already in the pool are added with rerank_score=0.0.
    """
    existing: dict[str, Candidate] = {c.citation: c for c in candidates}
    source_sets = {src: set(ranking) for src, ranking in source_rankings.items()}
    extracted_set: set[str] = source_sets.get("extracted", set())
    expected_codes: set[str] = (
        set(query_info.expected_codes) if query_info is not None else set()
    )

    # ── Collect seed parents and codes ────────────────────────────────────
    seeds = [c for c in candidates if _is_seed(c, seed_score_threshold)]
    seed_citations: set[str] = {c.citation for c in seeds}
    seed_parents: set[str] = set()
    seed_codes: set[str] = set()
    for c in seeds:
        p = _extract_bge_parent(c.citation)
        if p:
            seed_parents.add(p)
        code = _extract_law_code(c.citation)
        if code:
            seed_codes.add(code)

    # ── Mark same_parent / same_code on non-seed pool candidates ──────────
    for c in candidates:
        if c.citation in seed_citations:
            continue
        p = _extract_bge_parent(c.citation)
        if p and p in seed_parents:
            c.expansion_type = "same_parent"
            continue
        code = _extract_law_code(c.citation)
        if code and code in seed_codes:
            c.expansion_type = "same_code"

    # ── Source diversity rescue ───────────────────────────────────────────
    new_candidates: list[Candidate] = []
    for src, ranking in source_rankings.items():
        for citation in ranking[:rescue_top_k]:
            if citation in existing:
                continue
            sources = frozenset(s for s, ss in source_sets.items() if citation in ss)
            direct = citation in extracted_set
            code = _extract_law_code(citation)
            code_match = code is not None and code in expected_codes
            new_candidates.append(
                Candidate(
                    citation=citation,
                    rerank_score=0.0,
                    rrf_score=rrf_scores.get(citation, 0.0),
                    source_set=sources,
                    direct_regex_hit=direct,
                    expected_code_match=code_match,
                    expansion_type=f"{src}_rescue",
                )
            )
            existing[citation] = new_candidates[-1]  # prevent duplicate rescue

    return candidates + new_candidates


_VERIFIER_SYSTEM = """\
You are a Swiss legal citation relevance assessor.

Given a legal question and a list of candidate citations retrieved from Swiss legal \
databases, assess each candidate's relevance to the question.

For each candidate assign a relevance score:
  3 = directly answers the legal issue
  2 = useful supporting authority
  1 = weak or background relevance
  0 = unrelated or misleading

Provide a brief reason (1 sentence in English).

Also provide estimated_answer_count: how many total citations would be needed to \
fully answer the legal question. Simple questions need fewer citations (e.g. 3-5), \
questions with multiple independent legal sub-issues need more (e.g. 10-20).

STRICT RULES:
- You MUST score ALL candidates — the output list must have exactly the same count as input
- You MUST use ONLY the id values provided — never generate new ids
- estimated_answer_count must be a positive integer

Output exactly one valid JSON object matching this schema:
{
  "candidate_scores": [{"id": "...", "relevance": 0|1|2|3, "reason": "..."}],
  "estimated_answer_count": <integer>
}
"""


def _build_verifier_user_msg(
    query: str,
    query_info,
    items: list[dict],
) -> str:
    ctx: dict = {"question": query, "candidates": items}
    if query_info is not None:
        ctx["legal_issue"] = query_info.legal_issue
        ctx["expected_codes"] = query_info.expected_codes
        ctx["expected_articles"] = query_info.expected_articles
    return json.dumps(ctx, ensure_ascii=False)


def _apply_verifier_response(
    data: dict,
    id_to_candidate: dict[str, "Candidate"],
) -> int:
    """Parse LLM response, fill relevance into candidates. Returns n_LLM."""
    for item in data.get("candidate_scores", []):
        cid = item.get("id")
        if cid not in id_to_candidate:
            continue
        relevance = item.get("relevance")
        if not isinstance(relevance, int) or relevance not in (0, 1, 2, 3):
            continue
        c = id_to_candidate[cid]
        c.relevance = relevance
        c.llm_reason = str(item.get("reason", ""))

    raw = data.get("estimated_answer_count", 10)
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
        return 10
    return min(raw, 100)


def llm_verify(
    candidates: list[Candidate],
    query: str,
    query_info,
    corpus_texts: dict[str, str],
    *,
    verifier_top_k: int = 60,
    text_snippet_len: int = 300,
) -> tuple[list[Candidate], int]:
    """Score candidates with LLM. Returns (candidates, n_LLM).

    Fills relevance / llm_reason on the top verifier_top_k candidates (by
    rerank_score). Remaining candidates keep relevance=None.

    On LLM failure: retries once, then returns n_LLM=10 with all relevance=None.
    """
    top = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)[:verifier_top_k]
    id_to_candidate: dict[str, Candidate] = {f"C{i:03d}": c for i, c in enumerate(top)}

    items = [
        {
            "id": cid,
            "citation": c.citation,
            "text": (corpus_texts.get(c.citation) or "")[:text_snippet_len],
            "rerank_score": round(c.rerank_score, 4),
            "direct_hit": c.direct_regex_hit,
        }
        for cid, c in id_to_candidate.items()
    ]
    user_msg = _build_verifier_user_msg(query, query_info, items)

    for attempt in range(2):
        try:
            data = chat_json(_VERIFIER_SYSTEM, user_msg)
            n_llm = _apply_verifier_response(data, id_to_candidate)
            return candidates, n_llm
        except Exception as exc:
            print(f"[llm_verify] attempt {attempt + 1} failed: {exc}", file=sys.stderr)

    # Both attempts failed — return with all relevance=None
    print("[llm_verify] both attempts failed; using n_LLM=10", file=sys.stderr)
    return candidates, 10


_LLM_BOOST: dict[int, float] = {0: -0.40, 1: -0.10, 2: 0.15, 3: 0.35}


def fuse_scores(candidates: list[Candidate]) -> list[Candidate]:
    """Compute final_score for each candidate; return sorted descending.

    final_score = rerank_score + llm_relevance_boost + rule_boost
    """
    for c in candidates:
        llm_b = _LLM_BOOST.get(c.relevance, 0.0) if c.relevance is not None else 0.0

        rule_b = 0.0
        if c.direct_regex_hit:
            rule_b += 0.25
        if c.expected_code_match:
            rule_b += 0.06
        if len(c.source_set) >= 2:
            rule_b += 0.04
        if c.expansion_type == "same_parent":
            rule_b += 0.03
        elif c.expansion_type == "same_code":
            rule_b += 0.02
        if c.expansion_type is not None and c.expansion_type.endswith("_rescue"):
            rule_b -= 0.05
        if c.direct_regex_hit and c.expansion_type is not None:
            rule_b -= 0.05  # direct hit but low in reranker pool — partially offset

        c.final_score = c.rerank_score + llm_b + rule_b

    candidates.sort(key=lambda c: c.final_score, reverse=True)
    return candidates


def _elbow(scores: list[float]) -> int:
    """Return elbow point count using Kneedle algorithm (distance to min-max line)."""
    if len(scores) <= 2:
        return len(scores)
    from kneed import KneeLocator

    x = list(range(len(scores)))
    knl = KneeLocator(x, scores, curve="convex", direction="decreasing")
    knee = knl.knee
    return (knee + 1) if knee is not None else len(scores)


def adaptive_count(
    candidates: list[Candidate],
    n_llm: int,
    *,
    min_val: int = 3,
    max_val: int = 40,
) -> int:
    """Compute n_final = clamp(round(0.2*n_LLM + 0.4*n_elbow + 0.4*n_rel23), min, max).

    n_rel23 = count of candidates with relevance in (2, 3).
    candidates must already be sorted by final_score descending (fuse_scores output).
    """
    scores = [c.final_score for c in candidates]
    n_elbow = _elbow(scores)
    n_rel23 = sum(1 for c in candidates if c.relevance in (2, 3))
    raw = round(0.2 * n_llm + 0.4 * n_elbow + 0.4 * n_rel23)
    return max(min_val, min(max_val, raw))


def assemble(
    candidates: list[Candidate],
    n_final: int,
    *,
    max_per_parent: int = 3,
    rescue_score_floor: float = 0.1,
) -> list[str]:
    """Select final citation list from scored candidates.

    Steps (in order):
    1. Remove low-confidence rescue candidates (rerank_score < rescue_score_floor)
    2. Deduplicate (preserve final_score order)
    3. Take top n_final with BGE parent constraint (max_per_parent per BGE case)
    4. Append any direct_regex_hit candidates beyond n_final cutoff

    All output citations come from Candidate.citation — no new citations are generated.
    """
    # Step 1: filter low-quality rescue
    filtered = [
        c for c in candidates
        if not (
            c.expansion_type is not None
            and c.expansion_type.endswith("_rescue")
            and c.rerank_score < rescue_score_floor
        )
    ]

    # Step 2: dedup (keep highest final_score occurrence = first in sorted list)
    seen: set[str] = set()
    deduped: list[Candidate] = []
    for c in filtered:
        if c.citation not in seen:
            seen.add(c.citation)
            deduped.append(c)

    # Step 3 & 4: select with parent constraint; collect direct_hit overflow
    parent_counts: dict[str, int] = {}
    result: list[str] = []
    direct_hit_extras: list[str] = []

    for c in deduped:
        parent = _extract_bge_parent(c.citation)
        if parent and parent_counts.get(parent, 0) >= max_per_parent:
            continue

        if len(result) < n_final:
            result.append(c.citation)
            if parent:
                parent_counts[parent] = parent_counts.get(parent, 0) + 1
        elif c.direct_regex_hit:
            direct_hit_extras.append(c.citation)
            if parent:
                parent_counts[parent] = parent_counts.get(parent, 0) + 1

    return result + direct_hit_extras


def run_selector(
    query: str,
    query_info,                              # RewriteResult or None
    scored: list[tuple[str, float]],         # rerank_with_scores() output
    rrf_scores: dict[str, float],            # weighted_rrf() score dict
    source_rankings: dict[str, list[str]],   # per-path citation lists
    corpus_texts: dict[str, str],            # citation -> indexed_text
    *,
    # cheap_expand params
    rescue_top_k: int = 10,
    seed_score_threshold: float = 0.5,
    # llm_verify params
    use_llm_verify: bool = True,
    verifier_top_k: int = 60,
    text_snippet_len: int = 300,
    # adaptive_count params
    min_citations: int = 3,
    max_citations: int = 40,
    # assemble params
    max_per_parent: int = 3,
    rescue_score_floor: float = 0.1,
) -> list[str]:
    """Run the full selector pipeline and return the final citation list."""
    candidates = build_candidates(scored, rrf_scores, source_rankings, query_info)
    candidates = cheap_expand(
        candidates, source_rankings, rrf_scores, query_info,
        rescue_top_k=rescue_top_k,
        seed_score_threshold=seed_score_threshold,
    )

    n_llm = 10
    if use_llm_verify:
        candidates, n_llm = llm_verify(
            candidates, query, query_info, corpus_texts,
            verifier_top_k=verifier_top_k,
            text_snippet_len=text_snippet_len,
        )

    candidates = fuse_scores(candidates)
    n_final = adaptive_count(
        candidates, n_llm, min_val=min_citations, max_val=max_citations
    )
    return assemble(
        candidates, n_final,
        max_per_parent=max_per_parent,
        rescue_score_floor=rescue_score_floor,
    )
