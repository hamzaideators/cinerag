import os, json
from typing import List, Tuple, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "movies_vec")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
DOCS_JSON_PATH = os.getenv("DOCS_PATH", "movies_docs.json")

_client = QdrantClient(url=QDRANT_URL)
_model = SentenceTransformer(EMBED_MODEL)

def _load_text_map(path):
    try:
        docs = json.load(open(path))
        return {d["id"]: d.get("index_text","") for d in docs}
    except Exception:
        return {}

_text_map = _load_text_map(DOCS_JSON_PATH)

def qdrant_retrieve(
    query: str,
    k: int = 20,
    year: Optional[tuple] = None,
    genres: Optional[list] = None,
    return_text: bool = False,
    docs_json_path: Optional[str] = None,
) -> List[Tuple[float, str, str]]:
    """Return (score, tmdb_id, text_for_rerank) for top-k."""
    v = _model.encode([query], normalize_embeddings=True)[0]
    must = []
    if year:
        rng = {}
        if year[0] is not None: rng["gte"] = year[0]
        if year[1] is not None: rng["lte"] = year[1]
        must.append(FieldCondition(key="year", range=Range(**rng)))
    if genres:
        must.append(FieldCondition(key="genres", match={"any": genres}))
    flt = Filter(must=must) if must else None

    res = _client.search(collection_name=QDRANT_COLLECTION, query_vector=v, query_filter=flt, limit=k)
    text_map = _text_map if docs_json_path is None else _load_text_map(docs_json_path)
    results = []
    for r in res:
        mid = r.payload.get("tmdb_id")
        txt = text_map.get(mid, "") if return_text else ""
        results.append((float(r.score), mid, txt))
    return results
