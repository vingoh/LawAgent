"""Reciprocal Rank Fusion utilities."""


def weighted_rrf(
    rankings: list[tuple[list[str], float]],
    rrf_k: int = 60,
) -> list[str]:
    """Fuse multiple weighted rankings into a single ordered citation list.

    score(c) += weight / (rrf_k + rank + 1)  for each appearance in a ranking.
    """
    scores: dict[str, float] = {}
    for ranked_list, weight in rankings:
        if not ranked_list or weight == 0:
            continue
        for rank, citation in enumerate(ranked_list):
            scores[citation] = scores.get(citation, 0.0) + weight / (rrf_k + rank + 1)

    return sorted(scores, key=lambda c: scores[c], reverse=True)
