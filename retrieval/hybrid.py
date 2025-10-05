from typing import List, Tuple, Optional
from .es_search import es_retrieve
from .qdrant_search import qdrant_retrieve

def rrf_fuse(
    es_hits: List[Tuple[float, str, str]],
    qd_hits: List[Tuple[float, str, str]],
    k: int = 10,
    K: int = 60,
    prefer_text: bool = True,
) -> List[Tuple[float, str, str]]:
    """Reciprocal Rank Fusion of two ranked lists."""
    def to_rank_map(hits):
        return {mid: i+1 for i, (_, mid, _) in enumerate(hits)}

    r_es = to_rank_map(es_hits)
    r_qd = to_rank_map(qd_hits)

    text_map = {}
    if prefer_text:
        for s, mid, txt in es_hits + qd_hits:
            if txt and mid not in text_map:
                text_map[mid] = txt

    mids = set(r_es) | set(r_qd)
    fused = []
    for m in mids:
        score = 0.0
        if m in r_es: score += 1.0 / (K + r_es[m])
        if m in r_qd: score += 1.0 / (K + r_qd[m])
        fused.append((score, m, text_map.get(m, "")))
    fused.sort(key=lambda x: x[0], reverse=True)
    return fused[:k]

def hybrid_retrieve(
    query: str,
    k: int = 10,
    es_k: int = 30,
    qd_k: int = 30,
    year: Optional[tuple] = None,
    genres: Optional[list] = None,
    docs_json_path: Optional[str] = None,
) -> List[Tuple[float, str, str]]:
    es_hits = es_retrieve(query, k=es_k, year=year, genres=genres, return_text=True)
    qd_hits = qdrant_retrieve(query, k=qd_k, year=year, genres=genres, return_text=True, docs_json_path=docs_json_path)
    fused = rrf_fuse(es_hits, qd_hits, k=k, K=60, prefer_text=True)
    return fused
