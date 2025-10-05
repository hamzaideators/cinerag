"""
LLM client supporting multiple providers (OpenAI, Anthropic, vLLM).
"""

import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """Base LLM client interface."""

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(self, api_key: str = None, model: str = None, base_url: str = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url

        self.client = OpenAI(**kwargs)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using OpenAI API."""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 512)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content.strip()


class AnthropicClient(LLMClient):
    """Anthropic API client."""

    def __init__(self, api_key: str = None, model: str = None):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")

        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.client = Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Anthropic API."""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 512)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text.strip()


class VLLMClient(LLMClient):
    """vLLM server client (OpenAI-compatible endpoint)."""

    def __init__(self, base_url: str = None, model: str = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package not installed for vLLM client")

        self.base_url = base_url or os.getenv("VLLM_BASE_URL", "http://localhost:8001/v1")
        self.model = model or os.getenv("VLLM_MODEL", "gpt-3.5-turbo")

        self.client = OpenAI(
            base_url=self.base_url,
            api_key="EMPTY"  # vLLM doesn't require auth by default
        )

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using vLLM server."""
        temperature = kwargs.get("temperature", 0.7)
        max_tokens = kwargs.get("max_tokens", 512)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content.strip()


def get_llm_client(provider: str = None) -> Optional[LLMClient]:
    """
    Get LLM client based on provider.

    Args:
        provider: "openai", "anthropic", or "vllm". If None, uses LLM_PROVIDER env var.

    Returns:
        LLMClient instance or None if provider not configured.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "").lower()

    if not provider:
        return None

    if provider == "openai":
        try:
            return OpenAIClient()
        except (ImportError, ValueError) as e:
            print(f"OpenAI client not available: {e}")
            return None

    elif provider == "anthropic":
        try:
            return AnthropicClient()
        except (ImportError, ValueError) as e:
            print(f"Anthropic client not available: {e}")
            return None

    elif provider == "vllm":
        try:
            return VLLMClient()
        except ImportError as e:
            print(f"vLLM client not available: {e}")
            return None

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


def generate_answer(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    llm_client: LLMClient,
    **kwargs
) -> str:
    """
    Generate an answer using LLM based on query and retrieved documents.

    Args:
        query: User's query
        retrieved_docs: List of movie documents with metadata
        llm_client: LLM client instance
        **kwargs: Additional generation parameters

    Returns:
        Generated answer string
    """
    # Build context from retrieved documents
    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        title = doc.get("title", "Unknown")
        year = doc.get("year", "")
        year_str = f" ({year})" if year else ""
        overview = doc.get("overview") or doc.get("index_text", "No description available.")
        genres = ", ".join(doc.get("genres", []))

        context_parts.append(
            f"[{i}] {title}{year_str}\n"
            f"Genres: {genres}\n"
            f"Description: {overview[:300]}..."
        )

    context = "\n\n".join(context_parts)

    # Build prompt
    prompt = f"""You are a helpful movie recommendation assistant. Answer the user's query based on the provided movie information.

User Query: {query}

Available Movies:
{context}

Instructions:
- Provide a natural, conversational response
- Recommend the most relevant movie(s) from the list
- Explain why each movie matches the query
- Keep your response concise (2-3 sentences)
- Reference movies by their title

Answer:"""

    # Generate answer
    answer = llm_client.generate(prompt, **kwargs)

    return answer
