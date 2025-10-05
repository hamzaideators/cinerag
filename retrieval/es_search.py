import os
from typing import List, Tuple, Optional
from elasticsearch import Elasticsearch

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "movies_bm25")

_es = Elasticsearch(ES_URL)

def es_retrieve(
    query: str,
    k: int = 20,
    year: Optional[tuple] = None,
    genres: Optional[list] = None,
    return_text: bool = False,
) -> List[Tuple[float, str, str]]:
    """Return (score, tmdb_id, text_for_rerank) for top-k."""
    must = [{
        "multi_match": {
            "query": query,
            "fields": [
                "title^3",
                "keywords^2",
                "tagline^1.5",
                "overview^1",
                "reviews^0.75"
            ],
            "type": "best_fields",
            "operator": "or",
            "fuzziness": "AUTO"
        }
    }]
    if year:
        yfilter = {"range": {"year": {}}}
        if year[0] is not None:
            yfilter["range"]["year"]["gte"] = year[0]
        if year[1] is not None:
            yfilter["range"]["year"]["lte"] = year[1]
        must.append(yfilter)
    if genres:
        must.append({"terms": {"genres": genres}})

    body = {"query": {"bool": {"must": must}}}
    res = _es.search(index=ES_INDEX, size=k, query=body["query"])
    hits = res["hits"]["hits"]
    out = []
    for h in hits:
        src = h["_source"]
        text = ""
        if return_text:
            parts = [src.get("title",""), src.get("tagline",""), src.get("overview","")]
            text = " ".join([p for p in parts if p]).strip()
        out.append((float(h["_score"]), src.get("tmdb_id"), text))
    return out
