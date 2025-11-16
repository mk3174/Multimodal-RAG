import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

from ingest import ingest_text_file, ingest_image_file, ingest_audio_file

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from kg_builder import kg_client
except Exception as e:
    kg_client = None
    logger.warning(f"Could not import kg_builder.kg_client: {e}")

try:
    from langchain_rag import build_rag_chain, vectorstore
except Exception as e:
    build_rag_chain = None
    vectorstore = None
    logger.warning(f"Could not import langchain_rag: {e}")

# Global RAG runner instance
_rag_runner = None


def _ensure_rag_runner():
    """Ensure the global RAG runner is initialized."""
    global _rag_runner
    if _rag_runner is not None:
        return _rag_runner

    logger.info("Initializing RAG runner...")

    if build_rag_chain is None:
        raise RuntimeError("RAG builder is unavailable (langchain_rag import failed). Check configuration.")

    # Try to build with KG client for full hybrid capability
    try:
        if callable(build_rag_chain):
            _rag_runner = build_rag_chain(kg_client=kg_client)
            logger.info("RAG runner initialized with KG support")
            return _rag_runner
    except Exception as e:
        logger.warning(f"Failed to build RAG with KG client: {e}")

    # Fallback: build without KG (vector-only)
    try:
        if callable(build_rag_chain):
            _rag_runner = build_rag_chain(kg_client=None)
            logger.warning("RAG runner initialized WITHOUT KG support (vector-only mode)")
            return _rag_runner
    except Exception as e:
        logger.error(f"Failed to initialize RAG runner: {e}")
        raise RuntimeError(f"Cannot initialize RAG runner: {e}") from e


# ========== Ingest Functions (wrappers for Streamlit/API) ==========

def ingest_text(
    path: str,
    filename: Optional[str] = None,
    source_id: Optional[str] = None
) -> Dict[str, Any]:
    """Ingest a text file into vectorstore and knowledge graph.
    
    Args:
        path: Path to text/PDF file
        filename: Optional display filename
        source_id: Optional unique identifier (generated if not provided)
        
    Returns:
        Dict with status, source_id, and optional error info
    """
    if not Path(path).exists():
        error_msg = f"Text file not found: {path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    metadata = {
        "source_id": source_id or str(uuid.uuid4()),
        "filename": filename or Path(path).name
    }
    
    try:
        result = ingest_text_file(path, metadata)
        logger.info(f"Text file ingested successfully: {metadata['source_id']}")
        return result
    except Exception as e:
        logger.error(f"Text ingestion failed for {path}: {e}")
        raise


def ingest_image(
    path: str,
    filename: Optional[str] = None,
    source_id: Optional[str] = None
) -> Dict[str, Any]:
    """Ingest an image file (OCR + captioning).
    
    Args:
        path: Path to image file
        filename: Optional display filename
        source_id: Optional unique identifier
        
    Returns:
        Dict with status, source_id, and optional error info
    """
    if not Path(path).exists():
        error_msg = f"Image file not found: {path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    metadata = {
        "source_id": source_id or str(uuid.uuid4()),
        "filename": filename or Path(path).name
    }
    
    try:
        result = ingest_image_file(path, metadata)
        logger.info(f"Image file ingested successfully: {metadata['source_id']}")
        return result
    except Exception as e:
        logger.error(f"Image ingestion failed for {path}: {e}")
        raise


def ingest_audio(
    path: str,
    filename: Optional[str] = None,
    source_id: Optional[str] = None
) -> Dict[str, Any]:
    """Ingest an audio file (transcription).
    
    Args:
        path: Path to audio file
        filename: Optional display filename
        source_id: Optional unique identifier
        
    Returns:
        Dict with status, source_id, and optional error info
    """
    if not Path(path).exists():
        error_msg = f"Audio file not found: {path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    metadata = {
        "source_id": source_id or str(uuid.uuid4()),
        "filename": filename or Path(path).name
    }
    
    try:
        result = ingest_audio_file(path, metadata)
        logger.info(f"Audio file ingested successfully: {metadata['source_id']}")
        return result
    except Exception as e:
        logger.error(f"Audio ingestion failed for {path}: {e}")
        raise


# ========== Query Function ==========

def query_rag(query: str, top_k: int = 3) -> Dict[str, Any]:
    """Execute a RAG query.
    
    Args:
        query: User question (must be non-empty string)
        top_k: Maximum number of source documents to retrieve (1-20)
        
    Returns:
        Dict with 'answer' (str) and 'source_documents' (list)
    """
    # Validate inputs
    if not query or not isinstance(query, str):
        raise ValueError("`query` must be a non-empty string")
    
    if not isinstance(top_k, int) or top_k < 1:
        logger.warning(f"Invalid top_k={top_k}, using default=6")
        top_k = 6
    elif top_k > 20:
        logger.warning(f"top_k={top_k} too large, capping at 20")
        top_k = 20
    
    logger.info(f"Running RAG query with top_k={top_k}")
    
    try:
        runner = _ensure_rag_runner()
        result = runner.run(query, top_k=top_k)
        return result
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise RuntimeError(f"RAG query failed for '{query[:50]}...': {e}") from e


# ========== Utility Functions ==========

def reload_rag_runner(use_kg: bool = True) -> Any:
    """Rebuild the global RAG runner (useful during development).
    
    Args:
        use_kg: If True, attempt to use KG client; if False, build vector-only
        
    Returns:
        New RAG runner instance
    """
    global _rag_runner
    _rag_runner = None
    logger.info(f"Reloading RAG runner (use_kg={use_kg})...")
    
    try:
        if use_kg:
            return _ensure_rag_runner()
        else:
            _rag_runner = build_rag_chain(kg_client=None)
            logger.info("RAG runner reloaded in vector-only mode")
            return _rag_runner
    except Exception as e:
        logger.error(f"Failed to reload RAG runner: {e}")
        raise RuntimeError(f"Failed to rebuild RAG runner: {e}") from e


def get_system_status() -> Dict[str, Any]:
    """Get current system status (for debugging/monitoring).
    
    Returns:
        Dict with component status information
    """
    status: Dict[str, Any] = {
        "rag_runner_initialized": _rag_runner is not None,
        "kg_client_available": kg_client is not None,
        "vectorstore_available": vectorstore is not None
    }
    
    # Test KG connection if available
    if kg_client:
        try:
            # Simple test query
            kg_client.find_entities_by_text("test")
            status["kg_connection_ok"] = True
        except Exception as e:
            status["kg_connection_ok"] = False
            status["kg_error"] = str(e)
    
    return status
