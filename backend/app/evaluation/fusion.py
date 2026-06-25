from app.evaluation.models import FusedHit, RetrievalHit


def normalize_hit_scores(hits: list[RetrievalHit]) -> dict[str, float]:
    if not hits:
        return {}
    values = [hit.score for hit in hits]
    minimum = min(values)
    maximum = max(values)
    if maximum == minimum:
        score = 1.0 if maximum > 0 else 0.0
        return {hit.document_id: score for hit in hits}
    scale = maximum - minimum
    return {hit.document_id: (hit.score - minimum) / scale for hit in hits}


def fuse_hits(
    lexical: list[RetrievalHit],
    dense: list[RetrievalHit],
    *,
    bm25_weight: float = 0.4,
    dense_weight: float = 0.6,
    top_k: int | None = None,
) -> list[FusedHit]:
    if abs(bm25_weight + dense_weight - 1.0) > 1e-9:
        raise ValueError("retrieval weights must sum to 1")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be positive")

    lexical_scores = normalize_hit_scores(lexical)
    dense_scores = normalize_hit_scores(dense)
    document_ids = lexical_scores.keys() | dense_scores.keys()
    fused = [
        FusedHit(
            document_id=document_id,
            bm25_score=lexical_scores.get(document_id, 0.0),
            dense_score=dense_scores.get(document_id, 0.0),
            hybrid_score=(
                bm25_weight * lexical_scores.get(document_id, 0.0)
                + dense_weight * dense_scores.get(document_id, 0.0)
            ),
        )
        for document_id in document_ids
    ]
    fused.sort(key=lambda hit: (-hit.hybrid_score, hit.document_id))
    return fused if top_k is None else fused[:top_k]
