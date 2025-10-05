"""
LLM evaluation metrics using LLM-as-a-judge approach.
"""

from typing import List, Dict, Any
from llm import get_llm_client


def evaluate_relevance(query: str, answer: str, judge_client) -> float:
    """
    Evaluate if the answer is relevant to the query.
    Returns score from 0.0 to 1.0.
    """
    prompt = f"""You are an expert evaluator. Rate how relevant the answer is to the user's query.

Query: {query}

Answer: {answer}

Rate the relevance on a scale of 1-5:
1 = Not relevant at all
2 = Slightly relevant
3 = Moderately relevant
4 = Relevant
5 = Highly relevant and directly addresses the query

Provide ONLY a single number (1-5) as your response."""

    try:
        response = judge_client.generate(prompt, temperature=0.0)
        # Extract number from response
        score = float(response.strip().split()[0])
        # Normalize to 0-1
        return (score - 1) / 4.0
    except Exception as e:
        print(f"Relevance evaluation failed: {e}")
        return 0.5  # Default middle score


def evaluate_faithfulness(context: str, answer: str, judge_client) -> float:
    """
    Evaluate if the answer is faithful to the context (no hallucinations).
    Returns score from 0.0 to 1.0.
    """
    prompt = f"""You are an expert evaluator. Rate how faithful the answer is to the provided context.

Context:
{context}

Answer: {answer}

Rate the faithfulness on a scale of 1-5:
1 = Contains major hallucinations or contradictions
2 = Some information not supported by context
3 = Mostly grounded in context with minor additions
4 = Well grounded in context
5 = Completely faithful to context with no unsupported claims

Provide ONLY a single number (1-5) as your response."""

    try:
        response = judge_client.generate(prompt, temperature=0.0)
        score = float(response.strip().split()[0])
        return (score - 1) / 4.0
    except Exception as e:
        print(f"Faithfulness evaluation failed: {e}")
        return 0.5


def evaluate_coherence(answer: str, judge_client) -> float:
    """
    Evaluate if the answer is coherent and well-structured.
    Returns score from 0.0 to 1.0.
    """
    prompt = f"""You are an expert evaluator. Rate how coherent and well-structured the answer is.

Answer: {answer}

Rate the coherence on a scale of 1-5:
1 = Incoherent, fragmented, or nonsensical
2 = Somewhat coherent but poorly structured
3 = Moderately coherent with some flow issues
4 = Coherent and well-structured
5 = Excellent coherence, clarity, and structure

Provide ONLY a single number (1-5) as your response."""

    try:
        response = judge_client.generate(prompt, temperature=0.0)
        score = float(response.strip().split()[0])
        return (score - 1) / 4.0
    except Exception as e:
        print(f"Coherence evaluation failed: {e}")
        return 0.5


def evaluate_aspect_coverage(query: str, answer: str, expected_aspects: List[str], judge_client) -> float:
    """
    Evaluate if the answer covers the expected aspects from the query.
    Returns score from 0.0 to 1.0.
    """
    if not expected_aspects:
        return 1.0

    aspects_str = ", ".join(expected_aspects)

    prompt = f"""You are an expert evaluator. Check if the answer covers the expected aspects of the query.

Query: {query}

Expected aspects to cover: {aspects_str}

Answer: {answer}

For each expected aspect, determine if it's addressed in the answer.
Count how many aspects are covered and rate on scale of 1-5:
1 = None or almost none of the aspects covered
2 = Less than half covered
3 = About half covered
4 = Most aspects covered
5 = All aspects comprehensively covered

Provide ONLY a single number (1-5) as your response."""

    try:
        response = judge_client.generate(prompt, temperature=0.0)
        score = float(response.strip().split()[0])
        return (score - 1) / 4.0
    except Exception as e:
        print(f"Aspect coverage evaluation failed: {e}")
        return 0.5


def evaluate_answer(
    query: str,
    answer: str,
    context: str,
    expected_aspects: List[str] = None,
    judge_client = None
) -> Dict[str, float]:
    """
    Evaluate answer across multiple dimensions.

    Args:
        query: User query
        answer: Generated answer
        context: Context used to generate answer
        expected_aspects: List of aspects expected in answer
        judge_client: LLM client for evaluation

    Returns:
        Dictionary of metric scores
    """
    if judge_client is None:
        judge_client = get_llm_client()
        if judge_client is None:
            raise ValueError("No LLM client available for evaluation")

    metrics = {
        "relevance": evaluate_relevance(query, answer, judge_client),
        "faithfulness": evaluate_faithfulness(context, answer, judge_client),
        "coherence": evaluate_coherence(answer, judge_client),
    }

    if expected_aspects:
        metrics["aspect_coverage"] = evaluate_aspect_coverage(
            query, answer, expected_aspects, judge_client
        )

    # Compute overall score (average of all metrics)
    metrics["overall"] = sum(metrics.values()) / len(metrics)

    return metrics
