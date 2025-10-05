import os
from elasticsearch import Elasticsearch

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "movies_bm25")

es = Elasticsearch(ES_URL)

if es.indices.exists(index=ES_INDEX):
    es.indices.delete(index=ES_INDEX)

es.indices.create(
    index=ES_INDEX,
    settings={
        "analysis": {
            "analyzer": {
                "english_shingles": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase","kstem","porter_stem","shingle"]
                }
            }
        }
    },
    mappings={
        "properties": {
            "tmdb_id": {"type":"keyword"},
            "title":   {"type":"text","analyzer":"english","fields":{"kw":{"type":"keyword"}}},
            "tagline": {"type":"text","analyzer":"english"},
            "overview":{"type":"text","analyzer":"english"},
            "keywords":{"type":"text","analyzer":"english"},
            "reviews": {"type":"text","analyzer":"english"},
            "genres":  {"type":"keyword"},
            "year":    {"type":"integer"},
            "tmdb_url":{"type":"keyword"}
        }
    }
)
print("Elasticsearch index created:", ES_INDEX)
