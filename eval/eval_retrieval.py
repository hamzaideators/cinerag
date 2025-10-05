import os, json, argparse, statistics as stats
from typing import List
from retrieval.es_search import es_retrieve
from retrieval.qdrant_search import qdrant_retrieve
from retrieval.hybrid import hybrid_retrieve
from retrieval import reranker as rr
from eval.eval_metrics import recall_at_k, mrr, ndcg_at_k

def evaluate_backend(name: str, qrow: dict, k: int, docs_path: str):
    q = qrow["query"]; gold = qrow["gold"]
    if name == "es":
        hits = es_retrieve(q, k=k*5, return_text=False)
    elif name == "qdrant":
        hits = qdrant_retrieve(q, k=k*5, return_text=False, docs_json_path=docs_path)
    elif name == "hybrid":
        hits = hybrid_retrieve(q, k=k, es_k=k*5, qd_k=k*5, docs_json_path=docs_path)
    elif name == "hybrid_rerank":
        cands = hybrid_retrieve(q, k=max(50,k*5), es_k=max(50,k*5), qd_k=max(50,k*5), docs_json_path=docs_path)
        if rr.available():
            hits = rr.rerank(q, cands, top_k=k)
        else:
            raise RuntimeError("Reranker not available. Install torch + sentence-transformers and ensure the model can load.")
    else:
        raise ValueError(f"Unknown backend: {name}")
    ranked_ids = [h[1] for h in hits]
    return ranked_ids

def run(eval_path: str, backends: List[str], k: int, docs_path: str):
    lines = [json.loads(l) for l in open(eval_path)]
    results = []
    for backend in backends:
        rec5=[]; rec10=[]; mrrs=[]; ndcgs=[]
        for row in lines:
            ranked_ids = evaluate_backend(backend, row, k, docs_path)
            gold = row["gold"]
            rec5.append(recall_at_k(ranked_ids, gold, 5))
            rec10.append(recall_at_k(ranked_ids, gold, 10))
            mrrs.append(mrr(ranked_ids, gold))
            ndcgs.append(ndcg_at_k(ranked_ids, gold, k))
        summary = {
            "backend": backend,
            "queries": len(lines),
            "Recall@5": round(stats.mean(rec5), 3),
            "Recall@10": round(stats.mean(rec10), 3),
            "MRR": round(stats.mean(mrrs), 3),
            "nDCG@10": round(stats.mean(ndcgs), 3),
        }
        results.append(summary)
    winner = max(results, key=lambda r: (r["Recall@5"], r["MRR"]))
    return results, winner

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval/eval_queries.jsonl")
    ap.add_argument("--backends", nargs="+", default=["es","qdrant","hybrid","hybrid_rerank"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--docs", default="movies_docs.json")
    ap.add_argument("--out", default="reports/retrieval_results.json")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    results, winner = run(args.eval, args.backends, args.k, args.docs)
    print(json.dumps(results, indent=2))
    print("WINNER:", winner["backend"])
    with open(args.out, "w") as f:
        json.dump({"results": results, "winner": winner}, f, indent=2)
