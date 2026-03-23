"""
rag/embeddings.py
Gemini embeddings wrapper.
- Local dev: uses GOOGLE_API_KEY_LLM
- GCP: uses Vertex AI with Workload Identity (no API key needed)
"""

import os
import logging
from typing import List
from langchain_core.embeddings import Embeddings
from config.settings import settings

logger = logging.getLogger(__name__)


class GeminiEmbeddings(Embeddings):
    """
    Embedding model wrapper supporting both:
    - Local dev via google-genai SDK + API key
    - GCP via Vertex AI (set USE_VERTEX=true)
    """

    def __init__(self):
        self.model = settings.embed_model
        self._client = None

    @property
    def client(self):
        """Lazy-init client so it's not created at import time."""
        if self._client is None:
            use_vertex = os.environ.get("USE_VERTEX", "false").lower() == "true"
            if use_vertex:
                # On GCP — uses Workload Identity, no API key needed
                import vertexai
                from vertexai.language_models import TextEmbeddingModel
                vertexai.init(project=settings.gcp_project, location=settings.gcp_region)
                self._client = TextEmbeddingModel.from_pretrained(self.model)
                self._use_vertex = True
                logger.info(f"Using Vertex AI embeddings: {self.model}")
            else:
                # Local dev — uses google-genai SDK
                from google import genai
                self._client = genai.Client(api_key=settings.google_api_key_llm)
                self._use_vertex = False
                logger.info(f"Using Gemini API embeddings: {self.model}")
        return self._client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if getattr(self, "_use_vertex", False):
            embeddings = self.client.get_embeddings(texts)
            return [e.values for e in embeddings]
        else:
            from google.genai import types
            result = self.client.models.embed_content(
                model=self.model, contents=texts,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> List[float]:
        if getattr(self, "_use_vertex", False):
            embeddings = self.client.get_embeddings([text])
            return embeddings[0].values
        else:
            from google.genai import types
            result = self.client.models.embed_content(
                model=self.model, contents=[text],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
            )
            return result.embeddings[0].values
