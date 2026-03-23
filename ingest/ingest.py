"""
ingest/ingest.py
Document ingestion pipeline.
- Loads documents from LOCAL_DOCS_PATH
- Splits, embeds, builds FAISS index
- Saves locally and optionally uploads to GCS

Run: python ingest/ingest.py
"""

import sys
import logging
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(usecwd=True))

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from rag.embeddings import GeminiEmbeddings
from config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("=== COE Chatbot - Document Ingestion ===")

    # 1. Load documents
    docs_path = settings.local_docs_path
    logger.info(f"Loading documents from: {docs_path}")
    raw_documents = DirectoryLoader(docs_path).load()
    logger.info(f"  → {len(raw_documents)} document(s) found")

    if not raw_documents:
        logger.error("No documents found. Check LOCAL_DOCS_PATH in your .env")
        sys.exit(1)

    # 2. Split
    logger.info("Splitting into chunks...")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    ).split_documents(raw_documents)
    logger.info(f"  → {len(chunks)} chunks created")

    # 3. Embed + build index
    logger.info("Embedding chunks and building FAISS index...")
    embedding_model = GeminiEmbeddings()
    vector_store = FAISS.from_documents(chunks, embedding_model)

    # 4. Save locally
    local_path = settings.local_index_path
    Path(local_path).mkdir(parents=True, exist_ok=True)
    vector_store.save_local(local_path)
    logger.info(f"  → Index saved locally to: {local_path}")

    # 5. Upload to GCS if enabled
    if settings.use_gcs:
        logger.info(f"Uploading index to GCS: gs://{settings.gcs_bucket}/{settings.gcs_index_prefix}")
        from storage.gcs import upload_index
        upload_index(
            bucket_name=settings.gcs_bucket,
            gcs_prefix=settings.gcs_index_prefix,
            local_path=local_path,
        )
        logger.info("  → Upload complete")

    logger.info("✓ Ingestion complete!")


if __name__ == "__main__":
    main()
