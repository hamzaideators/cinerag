up:
	docker compose up -d

es-index:
	uv run python retrieval/es_setup.py
	uv run python retrieval/es_index.py

qdrant-upsert:
	uv run python retrieval/qdrant_upsert.py --docs movies_docs.json

eval:
	uv run python -m eval.eval_retrieval --eval eval/eval_queries.jsonl --docs movies_docs.json

eval-llm:
	uv run python -m eval.eval_llm --eval eval/eval_llm_queries.jsonl --docs movies_docs.json --backend hybrid_rerank --top-k 5

eval-all: eval eval-llm

api:
	docker compose up -d api ui

monitoring:
	docker compose up -d prometheus grafana

all:
	docker compose up -d

ingest:
	uv run python flows/tmdb_ingest.py --pages 500 --out movies_docs.json

.PHONY: up es-index qdrant-upsert eval eval-llm eval-all api monitoring all
