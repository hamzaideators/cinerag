import os, json, argparse, numpy as np
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="movies_docs.json")
    ap.add_argument("--url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    ap.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "movies_vec"))
    ap.add_argument("--model", default=os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2"))
    args = ap.parse_args()

    docs = json.load(open(args.docs))
    model = SentenceTransformer(args.model)

    texts = [d.get("index_text","") for d in docs]
    vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)

    client = QdrantClient(url=args.url)

    # (Re)create collection
    if args.collection in [c.name for c in client.get_collections().collections]:
        client.delete_collection(args.collection)

    client.recreate_collection(
        collection_name=args.collection,
        vectors_config=VectorParams(size=vecs.shape[1], distance=Distance.COSINE),
    )

    points = []
    for i, (doc, v) in enumerate(zip(docs, vecs)):
        payload = {
            "tmdb_id": doc["id"],
            "title": doc.get("title"),
            "genres": doc.get("genres", []),
            "year": doc.get("year"),
            "tmdb_url": doc.get("tmdb_url"),
        }
        points.append(PointStruct(id=i, vector=v.tolist(), payload=payload))

    # Batch upload
    B = 256
    for i in tqdm(range(0, len(points), B), desc="Upserting"):
        client.upsert(args.collection, points=points[i:i+B])

    # Index payload fields for filtering
    client.create_payload_index(args.collection, field_name="genres", field_schema="keyword")
    client.create_payload_index(args.collection, field_name="year", field_schema="integer")
    print("Qdrant collection ready:", args.collection, "points:", len(points))

if __name__ == "__main__":
    main()
