import os, json
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

ES_URL = os.getenv("ES_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ES_INDEX", "movies_bm25")
DOCS_PATH = os.getenv("DOCS_PATH", "movies_docs.json")

es = Elasticsearch(ES_URL)
docs = json.load(open(DOCS_PATH))

def to_es(doc):
    return {
        "_index": ES_INDEX,
        "_id": doc["id"],
        "_source": {
            "tmdb_id": doc["id"],
            "title": doc.get("title"),
            "tagline": doc.get("tagline"),
            "overview": doc.get("index_text", ""),
            "keywords": " ".join(doc.get("keywords", [])),
            "reviews": "",
            "genres": doc.get("genres", []),
            "year": doc.get("year"),
            "tmdb_url": doc.get("tmdb_url")
        }
    }

helpers.bulk(
    es,
    (to_es(d) for d in tqdm(docs, total=len(docs), desc="Indexing to ES", unit="doc"))
)
es.indices.refresh(index=ES_INDEX)
print("Indexed documents:", len(docs))
