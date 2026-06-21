"""Reciprocal Rank Fusion utilities."""


def weighted_rrf(
    rankings: list[tuple[list[str], float]],
    rrf_k: int = 60,
) -> tuple[list[str], dict[str, float]]:
    """Fuse multiple weighted rankings into a single ordered citation list.

    score(c) += weight / (rrf_k + rank + 1)  for each appearance in a ranking.

    Returns:
        (ordered_list, scores_dict) where scores_dict maps citation -> fused score.
    """
    scores: dict[str, float] = {}
    for ranked_list, weight in rankings:
        if not ranked_list or weight == 0:
            continue
        for rank, citation in enumerate(ranked_list):
            scores[citation] = scores.get(citation, 0.0) + weight / (rrf_k + rank + 1)

    ordered = sorted(scores, key=lambda c: scores[c], reverse=True)
    return ordered, scores
