# CineRAG Architecture Diagram

## System Architecture

```mermaid
graph TB
    subgraph External["External Services"]
        TMDB[("TMDB API<br/>Movie Data")]
        LLM[("LLM Providers<br/>OpenAI/Anthropic/vLLM")]
    end

    subgraph Ingestion["Data Ingestion"]
        Ingest["TMDB Ingest Pipeline<br/>flows/tmdb_ingest.py"]
        Docs[("movies_docs.json<br/>~10K Movies")]
    end

    subgraph Storage["Knowledge Bases"]
        ES[("Elasticsearch<br/>BM25 Index<br/>Port 9200")]
        Qdrant[("Qdrant<br/>Vector DB<br/>Port 6333")]
    end

    subgraph Retrieval["Retrieval Layer"]
        ESSearch["BM25 Search<br/>retrieval/es_search.py"]
        VecSearch["Vector Search<br/>retrieval/qdrant_search.py"]
        Hybrid["Hybrid RRF<br/>retrieval/hybrid.py"]
        Rerank["Reranker<br/>retrieval/reranker.py"]
    end

    subgraph Backend["Application Backend"]
        API["FastAPI Service<br/>app/main.py<br/>Port 8000"]
        LLMClient["LLM Client<br/>llm/client.py"]
    end

    subgraph Frontend["User Interface"]
        UI["Streamlit UI<br/>ui/app.py<br/>Port 8501"]
        User["👤 User"]
    end

    subgraph Monitoring["Monitoring Stack"]
        Prom["Prometheus<br/>Port 9090"]
        Grafana["Grafana<br/>Port 3000"]
        Metrics["Metrics:<br/>• Requests<br/>• Latency<br/>• Errors<br/>• Feedback"]
    end

    subgraph Evaluation["Evaluation"]
        EvalRet["Retrieval Eval<br/>eval/eval_retrieval.py"]
        EvalLLM["LLM Eval<br/>eval/eval_llm.py"]
        Reports["Reports<br/>reports/*.json"]
    end

    %% Data Ingestion Flow
    TMDB -->|"Fetch metadata,<br/>keywords, reviews"| Ingest
    Ingest -->|"Export JSON"| Docs
    Docs -->|"Index text"| ES
    Docs -->|"Embed & upload"| Qdrant

    %% Retrieval Flow
    API -->|"Query"| ESSearch
    API -->|"Query"| VecSearch
    API -->|"Query"| Hybrid
    ESSearch --> ES
    VecSearch --> Qdrant
    Hybrid --> ESSearch
    Hybrid --> VecSearch
    Hybrid -->|"Top candidates"| Rerank
    Rerank -->|"Re-ranked results"| API

    %% RAG Flow
    API -->|"Context + Query"| LLMClient
    LLMClient -->|"API call"| LLM
    LLM -->|"Response"| LLMClient
    LLMClient -->|"Answer"| API

    %% User Interaction
    User -->|"Natural language query"| UI
    UI -->|"POST /ask"| API
    API -->|"Answer + Citations"| UI
    UI -->|"Display results"| User
    UI -->|"POST /feedback<br/>(👍/👎)"| API

    %% Monitoring Flow
    API -->|"Export metrics"| Prom
    Prom -->|"Scrape /metrics"| API
    Grafana -->|"Query PromQL"| Prom
    Grafana -->|"Visualize dashboards"| Metrics

    %% Evaluation Flow
    Docs -->|"Ground truth"| EvalRet
    EvalRet -->|"Test queries"| Retrieval
    EvalLLM -->|"Test queries"| API
    EvalRet -->|"Metrics:<br/>Recall, MRR, nDCG"| Reports
    EvalLLM -->|"Metrics:<br/>Relevance, Faithfulness"| Reports

    %% Styling
    classDef external fill:#e1f5ff,stroke:#0288d1,stroke-width:2px
    classDef storage fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef service fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    classDef ui fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef monitoring fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef eval fill:#fff9c4,stroke:#f9a825,stroke-width:2px

    class TMDB,LLM external
    class ES,Qdrant,Docs storage
    class API,LLMClient,ESSearch,VecSearch,Hybrid,Rerank,Ingest service
    class UI,User ui
    class Prom,Grafana,Metrics monitoring
    class EvalRet,EvalLLM,Reports eval
```

## Component Details

### External Services
- **TMDB API**: Source of movie metadata, keywords, and reviews
- **LLM Providers**: OpenAI, Anthropic, or vLLM for answer generation

### Data Ingestion
- **TMDB Ingest Pipeline**: Fetches and enriches movie data
- **movies_docs.json**: Structured movie documents (~10K movies)

### Knowledge Bases
- **Elasticsearch**: BM25 keyword-based search index
- **Qdrant**: Vector database with semantic embeddings

### Retrieval Layer
- **BM25 Search**: Keyword matching via Elasticsearch
- **Vector Search**: Semantic similarity via Qdrant embeddings
- **Hybrid RRF**: Reciprocal Rank Fusion of BM25 + Vector
- **Reranker**: Cross-encoder model for final ranking refinement

### Application Backend
- **FastAPI Service**: RESTful API with `/ask` and `/feedback` endpoints
- **LLM Client**: Multi-provider LLM integration

### User Interface
- **Streamlit UI**: Interactive web interface with:
  - Natural language query input
  - Movie poster grid display
  - Feedback collection (👍/👎)
  - Backend and filter controls

### Monitoring Stack
- **Prometheus**: Metrics collection and storage
- **Grafana**: Real-time dashboards showing:
  - Request rates and latency
  - Error tracking
  - Backend usage distribution
  - User feedback sentiment

### Evaluation
- **Retrieval Evaluation**: Tests retrieval quality with Recall, MRR, nDCG
- **LLM Evaluation**: Assesses answer quality with Relevance, Faithfulness, Coherence

## Data Flows

### 1. Ingestion Flow
```
TMDB API → Ingest Pipeline → movies_docs.json → {Elasticsearch, Qdrant}
```

### 2. Query Flow
```
User → Streamlit UI → FastAPI → Retrieval Layer → {ES, Qdrant} → Reranker → LLM → Response
```

### 3. Feedback Flow
```
User → 👍/👎 → Streamlit UI → FastAPI /feedback → Prometheus → Grafana
```

### 4. Monitoring Flow
```
FastAPI → Prometheus /metrics → Prometheus → Grafana Dashboards
```

## Technology Stack

| Component | Technology | Port |
|-----------|-----------|------|
| Vector DB | Qdrant 1.9.0 | 6333 |
| Search Engine | Elasticsearch 8.14.1 | 9200 |
| Backend API | FastAPI + Uvicorn | 8000 |
| Frontend UI | Streamlit | 8501 |
| Metrics | Prometheus | 9090 |
| Dashboards | Grafana | 3000 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | - |
| Reranker | cross-encoder (ms-marco-MiniLM-L-6-v2) | - |

## Performance Metrics

### Retrieval Performance (5 test queries)
- **Hybrid + Rerank**: 80% Recall@5, 0.726 nDCG@10 (Winner)
- **Hybrid**: 60% Recall@5, 0.567 nDCG@10
- **Vector Only**: 40% Recall@5, 0.418 nDCG@10
- **BM25 Only**: 20% Recall@5, 0.26 nDCG@10

### LLM Answer Quality (10 test queries)
- **Overall Score**: 0.856 mean (0.938 median)
- **Relevance**: 0.925 (excellent query understanding)
- **Faithfulness**: 0.85 (minimal hallucinations)
- **Coherence**: 0.775 (good readability)
- **Aspect Coverage**: 0.875 (comprehensive answers)
