"""
rag/embeddings.py
Azure OpenAI embeddings wrapper.
- Replaces GeminiEmbeddings with AzureOpenAIEmbeddings
- Uses AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT from .env
"""

import os
import logging
from typing import List
from openai import AzureOpenAI
from langchain_core.embeddings import Embeddings
from config.settings import settings

logger = logging.getLogger(__name__)


class AzureEmbeddings(Embeddings):
    """
    Embedding model wrapper for Azure OpenAI.
    - Uses AzureOpenAI client with API key and endpoint
    - Compatible with LangChain Embeddings interface
    """

    def __init__(self):
        self.model = settings.azure_embedding_deployment
        self._client = None

    @property
    def client(self):
        """Lazy-init client so it's not created at import time."""
        if self._client is None:
            self._client = AzureOpenAI(
                api_key=settings.azure_openai_api_key,
                azure_endpoint=settings.azure_openai_endpoint,
                api_version=settings.azure_openai_api_version,
            )
            logger.info(f"Using Azure OpenAI embeddings: {self.model}")
        return self._client

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        try:
            logger.info(f"Embedding {len(texts)} document(s)...")
            response = self.client.embeddings.create(
                input=texts,
                model=self.model,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        try:
            response = self.client.embeddings.create(
                input=[text],
                model=self.model,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise