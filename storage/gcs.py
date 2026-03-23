"""
storage/gcs.py
Helpers for reading/writing to Google Cloud Storage.
Handles FAISS index sync and chat history persistence.
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_gcs_client():
    """Lazy import GCS client to avoid errors in local-only mode."""
    try:
        from google.cloud import storage
        return storage.Client()
    except ImportError:
        raise ImportError("google-cloud-storage not installed. Run: pip install google-cloud-storage")


def download_index(bucket_name: str, gcs_prefix: str, local_path: str) -> bool:
    """
    Download FAISS index files from GCS to local disk.
    Returns True if successful, False if index not found in GCS.
    """
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    local_path = Path(local_path)
    local_path.mkdir(parents=True, exist_ok=True)

    blobs = list(bucket.list_blobs(prefix=gcs_prefix))
    if not blobs:
        logger.warning(f"No index found at gs://{bucket_name}/{gcs_prefix}")
        return False

    for blob in blobs:
        filename = blob.name.replace(gcs_prefix + "/", "")
        if filename:
            dest = local_path / filename
            blob.download_to_filename(str(dest))
            logger.info(f"Downloaded {blob.name} → {dest}")

    return True


def upload_index(bucket_name: str, gcs_prefix: str, local_path: str):
    """Upload local FAISS index files to GCS."""
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    local_path = Path(local_path)

    for file in local_path.iterdir():
        blob = bucket.blob(f"{gcs_prefix}/{file.name}")
        blob.upload_from_filename(str(file))
        logger.info(f"Uploaded {file} → gs://{bucket_name}/{gcs_prefix}/{file.name}")


def save_chat_history(bucket_name: str, prefix: str, username: str, history: list):
    """Save a user's chat history as JSON to GCS."""
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{prefix}/{username}.json")
    blob.upload_from_string(json.dumps(history, indent=2), content_type="application/json")
    logger.info(f"Saved chat history for {username}")


def load_chat_history(bucket_name: str, prefix: str, username: str) -> list:
    """Load a user's chat history from GCS. Returns empty list if not found."""
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{prefix}/{username}.json")

    if not blob.exists():
        return []

    data = blob.download_as_text()
    return json.loads(data)
