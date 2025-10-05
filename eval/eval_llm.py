"""
LLM evaluation runner for CineRAG.
"""

import os
import json
import argparse
import statistics as stats
from typing import List
from tqdm import tqdm

from retrieval.hybrid import hybrid_retrieve
from retrieval import reranker as rr
from llm import get_llm_client, generate_answer
from eval.eval_llm_metrics import evaluate_answer


def load_movies_index(docs_path: str):
    """Load movies index for metadata lookup."""
    with open(docs_path, "r") as f:
        docs = json.load(f)

    index = {}
    for doc in docs:
        tmdb_id = doc.get("id") or doc.get("tmdb_id")
        if tmdb_id:
            index[tmdb_id] = doc

    return index


def evaluate_query(
    query_data: dict,
    movies_index: dict,
    llm_client,
    judge_client,
    backend: str = "hybrid_rerank",
    top_k: int = 5
):
    """
    Evaluate a single query.

    Args:
        query_data: Query dict with qid, query, gold, expected_aspects
        movies_index: Dictionary of movie documents
        llm_client: Client for generating answers
        judge_client: Client for judging answers
        backend: Retrieval backend to use
        top_k: Number of documents to retrieve

    Returns:
        Dictionary with evaluation results
    """
    query = query_data["query"]
    expected_aspects = query_data.get("expected_aspects", [])

    # Retrieve documents
    if backend == "hybrid_rerank":
        cands = hybrid_retrieve(query, k=max(50, top_k * 5))
        if rr.available():
            hits = rr.rerank(query, cands, top_k=top_k)
        else:
            hits = cands[:top_k]
    elif backend == "hybrid":
        hits = hybrid_retrieve(query, k=top_k)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Get retrieved documents
    retrieved_docs = []
    for score, mid, _ in hits:
        movie = movies_index.get(mid)
        if movie:
            retrieved_docs.append(movie)

    if not retrieved_docs:
        print(f"  WARNING: No documents retrieved for query: {query}")
        return None

    # Generate answer
    try:
        answer = generate_answer(query, retrieved_docs[:top_k], llm_client)
    except Exception as e:
        print(f"  ERROR generating answer: {e}")
        return None

    # Build context string for evaluation
    context_parts = []
    for doc in retrieved_docs[:top_k]:
        title = doc.get("title", "Unknown")
        year = doc.get("year", "")
        overview = doc.get("overview") or doc.get("index_text", "")
        context_parts.append(f"{title} ({year}): {overview[:200]}")

    context = "\n".join(context_parts)

    # Evaluate answer
    try:
        metrics = evaluate_answer(
            query=query,
            answer=answer,
            context=context,
            expected_aspects=expected_aspects,
            judge_client=judge_client
        )
    except Exception as e:
        print(f"  ERROR evaluating answer: {e}")
        return None

    return {
        "qid": query_data.get("qid"),
        "query": query,
        "answer": answer,
        "retrieved_count": len(retrieved_docs),
        "metrics": metrics
    }


def run_evaluation(
    eval_path: str,
    docs_path: str,
    backend: str = "hybrid_rerank",
    top_k: int = 5,
    out_path: str = "reports/llm_eval_results.json"
):
    """
    Run LLM evaluation on evaluation dataset.

    Args:
        eval_path: Path to eval queries JSONL file
        docs_path: Path to movies docs JSON file
        backend: Retrieval backend to use
        top_k: Number of documents to retrieve
        out_path: Path to save results
    """
    # Load evaluation queries
    with open(eval_path, "r") as f:
        queries = [json.loads(line) for line in f]

    print(f"Loaded {len(queries)} evaluation queries from {eval_path}")

    # Load movies index
    print(f"Loading movies index from {docs_path}...")
    movies_index = load_movies_index(docs_path)
    print(f"Loaded {len(movies_index)} movies")

    # Initialize LLM clients
    print("Initializing LLM clients...")
    llm_client = get_llm_client()
    if llm_client is None:
        raise ValueError("No LLM client available. Set LLM_PROVIDER and appropriate API key.")

    # Use same client for judging (could be different)
    judge_client = llm_client

    print(f"Using backend: {backend}, top_k: {top_k}")
    print("Starting evaluation...\n")

    # Run evaluation
    results = []
    for query_data in tqdm(queries, desc="Evaluating queries"):
        result = evaluate_query(
            query_data,
            movies_index,
            llm_client,
            judge_client,
            backend=backend,
            top_k=top_k
        )

        if result:
            results.append(result)
            # Print summary for this query
            metrics = result["metrics"]
            print(f"  QID {result['qid']}: "
                  f"Relevance={metrics['relevance']:.2f}, "
                  f"Faithfulness={metrics['faithfulness']:.2f}, "
                  f"Coherence={metrics['coherence']:.2f}, "
                  f"Overall={metrics['overall']:.2f}")

    # Compute aggregate statistics
    if not results:
        print("No results to aggregate!")
        return

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    metric_names = ["relevance", "faithfulness", "coherence", "aspect_coverage", "overall"]
    aggregates = {}

    for metric in metric_names:
        values = [r["metrics"].get(metric) for r in results if metric in r["metrics"]]
        if values:
            aggregates[metric] = {
                "mean": round(stats.mean(values), 3),
                "median": round(stats.median(values), 3),
                "stdev": round(stats.stdev(values), 3) if len(values) > 1 else 0.0,
                "min": round(min(values), 3),
                "max": round(max(values), 3)
            }
            print(f"{metric.capitalize():15s}: "
                  f"Mean={aggregates[metric]['mean']:.3f}, "
                  f"Median={aggregates[metric]['median']:.3f}, "
                  f"StdDev={aggregates[metric]['stdev']:.3f}")

    # Save results
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    output = {
        "config": {
            "eval_path": eval_path,
            "docs_path": docs_path,
            "backend": backend,
            "top_k": top_k,
            "num_queries": len(queries),
            "num_evaluated": len(results)
        },
        "aggregates": aggregates,
        "details": results
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM evaluation for CineRAG")
    parser.add_argument(
        "--eval",
        default="eval/eval_llm_queries.jsonl",
        help="Path to evaluation queries JSONL file"
    )
    parser.add_argument(
        "--docs",
        default="movies_docs.json",
        help="Path to movies docs JSON file"
    )
    parser.add_argument(
        "--backend",
        default="hybrid_rerank",
        choices=["hybrid", "hybrid_rerank"],
        help="Retrieval backend to use"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--out",
        default="reports/llm_eval_results.json",
        help="Output path for results"
    )

    args = parser.parse_args()

    run_evaluation(
        eval_path=args.eval,
        docs_path=args.docs,
        backend=args.backend,
        top_k=args.top_k,
        out_path=args.out
    )
