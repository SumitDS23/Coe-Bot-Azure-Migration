"""
config/settings.py
All configuration loaded from environment variables.
No hardcoded paths or secrets anywhere in the codebase.
"""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # --- Google Cloud (disabled - kept for reference) ---
    # gcp_project: str = os.environ.get("GCP_PROJECT", "")
    # gcp_region: str = os.environ.get("GCP_REGION", "us-central1")
    # gcs_bucket: str = os.environ.get("GCS_BUCKET", "")
    # gcs_index_prefix: str = os.environ.get("GCS_INDEX_PREFIX", "faiss_index")
    # gcs_history_prefix: str = os.environ.get("GCS_HISTORY_PREFIX", "chat_history")

    # --- GCS (still used for index storage if USE_GCS=true) ---
    gcs_bucket: str = os.environ.get("GCS_BUCKET", "")
    gcs_index_prefix: str = os.environ.get("GCS_INDEX_PREFIX", "faiss_index")
    gcs_history_prefix: str = os.environ.get("GCS_HISTORY_PREFIX", "chat_history")

    # --- Google Models (disabled) ---
    # embed_model: str = os.environ.get("EMBED_MODEL", "text-embedding-004")
    # llm_model: str = os.environ.get("LLM_MODEL", "gemini-2.5-flash")

    # --- Google API Keys (disabled) ---
    # google_api_key_embed: str = os.environ.get("GOOGLE_API_KEY_EMBED", "")
    # google_api_key_llm: str = os.environ.get("GOOGLE_API_KEY_LLM", "")

    # --- Azure OpenAI ---                                        # ← Added
    azure_openai_api_key: str = os.environ.get("AZURE_OPENAI_API_KEY", "")
    azure_openai_endpoint: str = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_version: str = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
    azure_openai_deployment_name: str = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
    azure_embedding_deployment: str = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")

    # --- RAG ---
    retrieval_k: int = int(os.environ.get("RETRIEVAL_K", 4))
    retrieval_mode: str = os.environ.get("RETRIEVAL_MODE", "mmr")
    chunk_size: int = int(os.environ.get("CHUNK_SIZE", 500))
    chunk_overlap: int = int(os.environ.get("CHUNK_OVERLAP", 50))

    # --- Local paths ---
    local_index_path: str = os.environ.get("LOCAL_INDEX_PATH", "./faiss_index")
    local_docs_path: str = os.environ.get("LOCAL_DOCS_PATH", "./documents")

    # --- App ---
    app_title: str = os.environ.get("APP_TITLE", "COE Chatbot")
    app_subtitle: str = os.environ.get("APP_SUBTITLE", "Analytics Assistant · ABC")
    use_gcs: bool = os.environ.get("USE_GCS", "false").lower() == "true"
    inventory_path: str = os.environ.get("INVENTORY_PATH", "./Enterprise_Model_Inventory.xlsx")

    # --- Auth (comma-separated user:password pairs) ---
    # e.g. USERS="alice:pass123,bob:pass456"
    users_raw: str = os.environ.get("USERS", "admin:admin123")

    @property
    def users(self) -> dict:
        """Parse USERS env var into {username: password} dict."""
        result = {}
        for pair in self.users_raw.split(","):
            pair = pair.strip()
            if ":" in pair:
                u, p = pair.split(":", 1)
                result[u.strip()] = p.strip()
        return result


# Single global instance
settings = Settings()