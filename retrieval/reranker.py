import os
from typing import List, Tuple
try:
    from sentence_transformers import CrossEncoder
    _ce = CrossEncoder(os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
except Exception as e:
    _ce = None
    _err = e

def available() -> bool:
    return _ce is not None

def rerank(query: str, candidates: List[Tuple[float, str, str]], top_k: int = 10) -> List[Tuple[float, str, str]]:
    """Rescore (score,id,text) candidates with a cross-encoder."""
    if _ce is None:
        raise RuntimeError(f"Reranker model not available: {_err}")
    pairs = [[query, c[2] or ""] for c in candidates]
    scores = _ce.predict(pairs)
    rescored = [(float(s), candidates[i][1], candidates[i][2]) for i, s in enumerate(scores)]
    rescored.sort(key=lambda x: x[0], reverse=True)
    return rescored[:top_k]
