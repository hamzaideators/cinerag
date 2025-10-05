import os, time, json
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from retrieval.es_search import es_retrieve
from retrieval.qdrant_search import qdrant_retrieve
from retrieval.hybrid import hybrid_retrieve
from retrieval import reranker as rr
from llm import get_llm_client, generate_answer

APP_PORT = int(os.getenv("APP_PORT", "8000"))
DOCS_PATH = os.getenv("DOCS_PATH", "movies_docs.json")
FEEDBACK_LOG_PATH = os.getenv("FEEDBACK_LOG_PATH", "feedback.jsonl")

app = FastAPI(title="CineRAG API", version="0.1.0")

# Initialize default LLM client from .env (may be None if not configured)
llm_client = get_llm_client()

# Cache for dynamically requested LLM clients
_LLM_CLIENTS_CACHE = {}

def get_llm_client_for_provider(provider: Optional[str] = None):
    """
    Get or create LLM client for specified provider.

    Args:
        provider: Provider name (openai, anthropic, vllm) or None for default

    Returns:
        LLM client instance or None
    """
    # If no provider specified, use default from .env
    if not provider or provider == "auto":
        return llm_client

    # Check cache
    if provider in _LLM_CLIENTS_CACHE:
        return _LLM_CLIENTS_CACHE[provider]

    # Create new client for this provider and cache it
    try:
        client = get_llm_client(provider)
        _LLM_CLIENTS_CACHE[provider] = client
        return client
    except Exception as e:
        print(f"Failed to create {provider} client: {e}")
        return None

# Load movies index for metadata lookup
_MOVIES_INDEX = {}
try:
    with open(DOCS_PATH, "r") as f:
        docs = json.load(f)
        for doc in docs:
            tmdb_id = doc.get("id") or doc.get("tmdb_id")
            if tmdb_id:
                _MOVIES_INDEX[tmdb_id] = doc
    print(f"Loaded {len(_MOVIES_INDEX)} movies from {DOCS_PATH}")
except Exception as e:
    print(f"Warning: Could not load movies index: {e}")

# ---- Prometheus metrics ----
REQS = Counter("cinerag_requests_total", "Total requests", ["endpoint"])
ERRORS = Counter("cinerag_errors_total", "Errors", ["endpoint", "type"])
STAGE_LAT = Histogram("cinerag_stage_latency_seconds", "Latency per stage", ["stage"])
BACKEND_USED = Counter("cinerag_backend_requests_total", "Backend usage", ["backend"])
FEEDBACK = Counter("cinerag_feedback_total", "User feedback", ["thumb"])

class AskRequest(BaseModel):
    query: str
    top_k: int = 7
    backend: str = "auto"   # es | qdrant | hybrid | hybrid_rerank | auto
    provider: Optional[str] = None  # openai | anthropic | vllm | None (uses .env default)
    year: Optional[List[Optional[int]]] = None  # [start, end]
    genres: Optional[List[str]] = None

class Citation(BaseModel):
    tmdb_id: str
    title: Optional[str] = None
    year: Optional[int] = None
    url: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    retrieved: List[Dict[str, Any]]
    backend: str

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    citations: List[str] = []
    thumb: str  # "up" or "down"
    comment: Optional[str] = None

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/metrics")
def metrics():
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)

def _retrieve(query: str, top_k: int, backend: str, year=None, genres=None, docs_path="movies_docs.json"):
    # returns list of dicts with id and score
    start = time.time()
    hits = []
    if backend == "es":
        hits = es_retrieve(query, k=top_k, year=year, genres=genres, return_text=False)
    elif backend == "qdrant":
        hits = qdrant_retrieve(query, k=top_k, year=year, genres=genres, return_text=False, docs_json_path=docs_path)
    elif backend == "hybrid":
        hits = hybrid_retrieve(query, k=top_k, year=year, genres=genres, docs_json_path=docs_path)
    elif backend == "hybrid_rerank":
        cands = hybrid_retrieve(query, k=max(50, top_k*5), year=year, genres=genres, docs_json_path=docs_path)
        if rr.available():
            hits = rr.rerank(query, cands, top_k=top_k)
        else:
            raise HTTPException(500, "Reranker model not available")
    else:
        raise HTTPException(400, f"Unknown backend {backend}")
    STAGE_LAT.labels(stage="retrieve").observe(time.time()-start)
    return hits

def _auto_backend():
    # simple rule: prefer hybrid_rerank if available, else hybrid
    if rr.available():
        return "hybrid_rerank"
    return "hybrid"

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    REQS.labels(endpoint="/ask").inc()
    backend = req.backend if req.backend != "auto" else _auto_backend()
    BACKEND_USED.labels(backend=backend).inc()
    year = tuple(req.year) if req.year else None

    # Retrieve relevant documents
    try:
        hits = _retrieve(req.query, req.top_k, backend, year=year, genres=req.genres)
    except HTTPException as e:
        ERRORS.labels(endpoint="/ask", type="retrieve").inc()
        raise e

    # Build citations and retrieved list
    citations = []
    retrieved = []
    retrieved_docs = []

    for score, mid, _ in hits:
        # Get movie metadata from index
        movie = _MOVIES_INDEX.get(mid, {})
        title = movie.get("title")
        movie_year = movie.get("year")
        url = movie.get("tmdb_url")

        citations.append({
            "tmdb_id": mid,
            "title": title,
            "year": movie_year,
            "url": url
        })
        retrieved.append({"score": score, "tmdb_id": mid})

        # Collect full docs for LLM context
        if movie:
            retrieved_docs.append(movie)

    # Get LLM client for requested provider (or default)
    selected_llm_client = get_llm_client_for_provider(req.provider)

    # Generate answer using LLM if available
    if selected_llm_client and retrieved_docs:
        try:
            start = time.time()
            answer = generate_answer(req.query, retrieved_docs[:5], selected_llm_client)
            STAGE_LAT.labels(stage="llm").observe(time.time() - start)
        except Exception as e:
            print(f"LLM generation failed: {e}")
            ERRORS.labels(endpoint="/ask", type="llm").inc()
            # Fallback to simple answer
            answer = f"Based on your query, I found these relevant movies: " + ", ".join([
                c["title"] or c["tmdb_id"] for c in citations[:3]
            ])
    else:
        # Fallback when LLM not configured
        answer = f"Based on your query, I found these relevant movies: " + ", ".join([
            c["title"] or c["tmdb_id"] for c in citations[:3]
        ])

    return AskResponse(answer=answer, citations=citations, retrieved=retrieved, backend=backend)

def log_feedback_to_file(feedback_data: dict):
    """
    Append feedback to JSON Lines file for persistent storage.
    Fails silently if file write errors occur.
    """
    try:
        # Ensure parent directory exists
        import pathlib
        pathlib.Path(FEEDBACK_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)

        with open(FEEDBACK_LOG_PATH, "a") as f:
            json.dump(feedback_data, f)
            f.write("\n")
    except Exception as e:
        print(f"Warning: Could not write feedback to {FEEDBACK_LOG_PATH}: {e}")

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    REQS.labels(endpoint="/feedback").inc()
    if req.thumb not in ("up","down"):
        ERRORS.labels(endpoint="/feedback", type="bad_thumb").inc()
        raise HTTPException(400, "thumb must be 'up' or 'down'")

    FEEDBACK.labels(thumb=req.thumb).inc()

    # Log feedback to persistent storage
    feedback_data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "query": req.query,
        "answer": req.answer,
        "citations": req.citations,
        "thumb": req.thumb,
        "comment": req.comment or ""
    }
    log_feedback_to_file(feedback_data)

    return {"ok": True}
