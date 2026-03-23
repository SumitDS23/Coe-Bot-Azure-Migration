"""
config/settings.py
All configuration loaded from environment variables.
No hardcoded paths or secrets anywhere in the codebase.
"""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    # --- Google Cloud ---
    gcp_project: str = os.environ.get("GCP_PROJECT", "")
    gcp_region: str = os.environ.get("GCP_REGION", "us-central1")
    gcs_bucket: str = os.environ.get("GCS_BUCKET", "")
    gcs_index_prefix: str = os.environ.get("GCS_INDEX_PREFIX", "faiss_index")
    gcs_history_prefix: str = os.environ.get("GCS_HISTORY_PREFIX", "chat_history")

    # --- Models ---
    embed_model: str = os.environ.get("EMBED_MODEL", "text-embedding-004")
    llm_model: str = os.environ.get("LLM_MODEL", "gemini-2.5-flash")

    # --- API Keys (local dev only, use Workload Identity on GCP) ---
    google_api_key_embed: str = os.environ.get("GOOGLE_API_KEY_EMBED", "")
    google_api_key_llm: str = os.environ.get("GOOGLE_API_KEY_LLM", "")

    # --- RAG ---
    retrieval_k: int = int(os.environ.get("RETRIEVAL_K", 4))
    retrieval_mode: str = os.environ.get("RETRIEVAL_MODE", "mmr")
    chunk_size: int = int(os.environ.get("CHUNK_SIZE", 500))
    chunk_overlap: int = int(os.environ.get("CHUNK_OVERLAP", 50))

    # --- Local paths (used in local dev, overridden on GCP) ---
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
