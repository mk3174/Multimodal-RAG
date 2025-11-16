import os
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from langchain_qdrant import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


try:
    from kg_builder import kg_client  # previous file's KGClient instance
except Exception:
    kg_client = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
LLM_MODEL = os.getenv("LLM_MODEL")



def init_qdrant_and_vectorstore(
    collection_name: str = QDRANT_COLLECTION,
    vector_size: int = 1536,
    distance: str = "Cosine"
    ) -> Tuple[QdrantClient, Qdrant]:
    """
    Initialize Qdrant client and LangChain Qdrant vectorstore wrapper.
    If collection doesn't exist, create it.
    Returns (qdrant_client, langchain_qdrant_vectorstore)
    """
    logger.info(f"Connecting to Qdrant at {QDRANT_URL}")

    client = QdrantClient(
        url=f"https://{QDRANT_URL}",
        api_key=QDRANT_API_KEY or None,
        prefer_grpc=False
    )

    vsz = vector_size

    try:
        collections_response = client.get_collections()
        existing_collections = [c.name for c in collections_response.collections]
        if collection_name not in existing_collections:
            logger.info(f"Creating Qdrant collection '{collection_name}' (dim={vsz})")
            client.recreate_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=vsz, distance=qmodels.Distance[distance.upper()])
            )
        else:
            logger.info(f"Qdrant collection '{collection_name}' already exists")
    except Exception as e:
        # some qdrant-client versions return different shapes from get_collections; use safe creation
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(size=vsz, distance=qmodels.Distance[distance.upper()])
            )
        except Exception:
            logger.info("Collection create/recreate attempted; continuing if exists")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)

    # LangChain Qdrant wrapper - uses qdrant_client under the hood
    vectorstore = Qdrant(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings
    )

    return client, vectorstore


# Initialize once (lazy)
_qdrant_client: Optional[QdrantClient] = None
_vectorstore: Optional[Qdrant] = None

def ensure_vectorstore() -> Qdrant:
    global _qdrant_client, _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    _qdrant_client, _vectorstore = init_qdrant_and_vectorstore()
    return _vectorstore


# ========== Utilities: chunking and ID generation ==========
def _gen_id(prefix: str = "doc") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Simple sliding-window chunker"""
    if not text:
        return []
    text = text.strip()
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        start = end - overlap if end < L else end
    return chunks


# ========== Ingestors (text / image / audio) ==========
# These ingestors extract text, create LangChain Documents with metadata, embed & upsert into Qdrant,
# and (optionally) add Document node to Neo4j KG.

def ingest_text_file(path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ingests a text file (txt, or PDF pre-extracted as text).
    - metadata must contain at least an 'id' field (unique)
    - reads file, chunks text, builds Documents with metadata, and adds to vectorstore & KG
    """
    pathp = Path(path)
    if not pathp.exists():
        raise FileNotFoundError(path)

    ensure_vectorstore()
    with open(pathp, "r", encoding="utf-8", errors="ignore") as fh:
        raw = fh.read()

    # Chunk and create LangChain Documents
    chunks = chunk_text(raw)
    docs: List[Document] = []
    base_id = metadata.get("id") or _gen_id("doc")
    for i, chunk in enumerate(chunks):
        doc_meta = metadata.copy()
        doc_meta.update({"chunk_index": i, "source": str(pathp.name)})
        docs.append(Document(page_content=chunk, metadata=doc_meta))

    # Upsert to vectorstore
    vs = ensure_vectorstore()
    vs.add_documents(docs)
    logger.info(f"Upserted {len(docs)} chunks from {path} into Qdrant collection '{QDRANT_COLLECTION}'")

    # Add to KG if available: create Document node with metadata (top-level only)
    if kg_client:
        try:
            # Ensure the node 'id' is present: align with metadata key 'id'
            kg_client.add_document(base_id, metadata)
        except Exception as e:
            logger.warning(f"Failed to add doc to KG: {e}")

    return {"status": "ok", "id": base_id, "chunks": len(docs)}


def ingest_image_file(path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ingest an image file:
    - performs OCR (if pytesseract available) OR uses a placeholder caption
    - chunks (if long) and upserts to vectorstore + KG
    """
    pathp = Path(path)
    if not pathp.exists():
        raise FileNotFoundError(path)

    # Try to OCR if pytesseract is installed
    text = None
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(pathp)
        text = pytesseract.image_to_string(img)
    except Exception:
        # fallback: simple filename-as-caption
        text = f"[Image content: {pathp.name}]"

    # reuse text ingestion logic for simplicity
    metadata = metadata.copy()
    metadata.setdefault("id", metadata.get("id") or _gen_id("img"))
    metadata.setdefault("mime_type", "image")
    metadata.setdefault("filename", pathp.name)

    # chunk the OCR/caption text
    chunks = chunk_text(text)
    docs = []
    for i, chunk in enumerate(chunks or [text]):
        doc_meta = metadata.copy()
        doc_meta.update({"chunk_index": i, "source": pathp.name})
        docs.append(Document(page_content=chunk, metadata=doc_meta))

    vs = ensure_vectorstore()
    vs.add_documents(docs)
    logger.info(f"Upserted image-derived {len(docs)} docs for {path} into Qdrant")

    if kg_client:
        try:
            kg_client.add_document(metadata["id"], metadata)
        except Exception as e:
            logger.warning(f"Failed to add image doc to KG: {e}")

    return {"status": "ok", "id": metadata["id"], "chunks": len(docs)}


def ingest_audio_file(path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:

    pathp = Path(path)
    if not pathp.exists():
        raise FileNotFoundError(path)

    transcript = None
    try:
        import whisper
        model = whisper.load_model("small")
        res = model.transcribe(str(pathp))
        transcript = res["text"]
    except Exception:
        transcript = f"[Audio: {pathp.name} -- transcription not available locally]"
    

    metadata = metadata.copy()
    metadata.setdefault("id", metadata.get("id") or _gen_id("aud"))
    metadata.setdefault("mime_type", "audio")
    metadata.setdefault("filename", pathp.name)

    # Ensure transcript is a string
    transcript_text: str = transcript if isinstance(transcript, str) else (transcript[0] if isinstance(transcript, list) and transcript else "")
    
    chunks = chunk_text(transcript_text)
    docs = []
    for i, chunk in enumerate(chunks or [transcript_text]):
        doc_meta = metadata.copy()
        doc_meta.update({"chunk_index": i, "source": pathp.name})
        docs.append(Document(page_content=chunk, metadata=doc_meta))

    vs = ensure_vectorstore()
    vs.add_documents(docs)
    logger.info(f"Upserted audio-derived {len(docs)} docs for {path} into Qdrant")

    if kg_client:
        try:
            kg_client.add_document(metadata["id"], metadata)
        except Exception as e:
            logger.warning(f"Failed to add audio doc to KG: {e}")

    return {"status": "ok", "id": metadata["id"], "chunks": len(docs)}


# ========== Hybrid Search = KG traversal + metadata filter + vector search ==========
def hybrid_retrieve(
    query: str,
    top_k: int = 6,
    use_kg: bool = True,
    kg_entity: Optional[str] = None,
    kg_hops: int = 1,
    metadata_filters: Optional[Dict[str, Any]] = None
) -> List[Document]:
    """
    Hybrid retrieval pipeline:
      1) If use_kg and kg_entity provided: fetch neighbor docs via KG traversal (ids)
      2) Use metadata_filters to reduce candidate space (e.g., {"mime_type": "pdf"})
      3) Run vector search in Qdrant with filter on candidate IDs if present
      4) Merge results and return top_k Documents (LangChain Document objects)
    """
    vs = ensure_vectorstore()
    embeddings = vs.embeddings  # OpenAIEmbeddings used during init

    # 1) KG neighbors -> collect candidate IDs (document ids saved in KG nodes as 'id')
    candidate_ids = None
    if use_kg and kg_entity and kg_client:
        try:
            neighbor_docs = kg_client.get_neighbor_documents(kg_entity, hops=kg_hops, limit=200)
            # neighbor_docs are LangChain Documents (per earlier KG code) with metadata that include id
            candidate_ids = {d.metadata.get("id") for d in neighbor_docs if d.metadata.get("id")}
            logger.info(f"KG neighbors found {len(candidate_ids)} candidates")
        except Exception as e:
            logger.warning(f"KG neighbor retrieval failed: {e}")
            candidate_ids = None

    # Build Qdrant filter (payload filter) - Qdrant uses 'payload' naming for metadata
    # LangChain Qdrant wrapper accepts 'filter' as a list/dict consistent with qdrant-client.
    qdrant_filter = {}
    if metadata_filters:
        # convert to simple must filter
        # NOTE: the exact filter shape depends on qdrant-client version; we rely on langchain wrapper passing through
        qdrant_filter["must"] = [{"key": k, "match": {"value": v}} for k, v in metadata_filters.items()]

    if candidate_ids:
        # filter by ids (payload field 'id' matching)
        # Qdrant supports filtering by payload, so we add an 'in' style filter
        # Implementation note: LangChain's Qdrant similarity_search has a `filter` param expecting a dict.
        # We'll send a generic filter with must+should combining id matches.
        id_filters = [{"key": "id", "match": {"value": _id}} for _id in list(candidate_ids)[:500]]
        if "must" in qdrant_filter:
            qdrant_filter["must"].extend(id_filters)
        else:
            qdrant_filter["must"] = id_filters

    # 2) Vector search
    query_embedding = embeddings.embed_query(query)
    # LangChain's Qdrant wrapper supports similarity_search_by_vector with filter param
    try:
        raw_results = vs.similarity_search_by_vector(query_embedding, k=top_k, filter=qdrant_filter or None)
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raw_results = []

    # raw_results is list[Document] already (LangChain wrapper)
    # 3) If no KG candidate_ids provided, optionally fall back to plain vector results
    results = raw_results

    # 4) De-duplicate by metadata id and return final list
    seen_ids = set()
    deduped = []
    for doc in results:
        mid = doc.metadata.get("id")
        if mid and mid in seen_ids:
            continue
        seen_ids.add(mid)
        deduped.append(doc)
        if len(deduped) >= top_k:
            break

    return deduped


# ========== RAG Runner ==========
class RAGRunner:
    """
    Simple RAG runner that:
      - retrieves documents via hybrid_retrieve()
      - creates a prompt for the LLM with retrieved documents as context
      - returns a dictionary with answer and source_documents
    """
    def __init__(self, llm_model: str = LLM_MODEL, temperature: float = 0.0):
        if OPENAI_API_KEY is None:
            logger.warning("OPENAI_API_KEY not set; LLM will not be available")
            self.llm = None
        else:
            self.llm = ChatOpenAI(model=llm_model, temperature=temperature, openai_api_key=OPENAI_API_KEY)

    def run(self, query: str, top_k: int = 6, use_kg: bool = True, kg_entity: Optional[str] = None) -> Dict[str, Any]:
        docs = hybrid_retrieve(query, top_k=top_k, use_kg=use_kg, kg_entity=kg_entity)
        # Build context string
        context_parts = []
        sources = []
        for d in docs:
            # include chunk index and source in metadata for traceability
            mid = d.metadata.get("id", "unknown")
            chunk_idx = d.metadata.get("chunk_index", None)
            source_name = d.metadata.get("filename") or d.metadata.get("source")
            context_parts.append(f"---\nSource: {source_name}\nDocID: {mid}\nChunk: {chunk_idx}\n\n{d.page_content}\n")
            sources.append({"id": mid, "metadata": d.metadata})

        context_text = "\n".join(context_parts) or "[NO_CONTEXT]"

        system_prompt = (
            "You are a helpful assistant. Use the provided context snippets to answer the user's question. "
            "Cite the source IDs in your answer when appropriate."
        )

        user_prompt = f"QUESTION: {query}\n\nCONTEXT:\n{context_text}\n\nProvide a concise answer and list source IDs used."

        # Query LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        try:
            if self.llm is None:
                answer = "Error: LLM not available (OPENAI_API_KEY not set)."
            else:
                llm_resp = self.llm.invoke(messages)
                answer = (llm_resp.content or "").strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            answer = "Error: LLM call failed."

        return {"answer": answer, "source_documents": sources}


# convenience: build and return a RAG runner (optionally pass kg_client)
_rag_runner: Optional[RAGRunner] = None

def build_rag_runner(force_rebuild: bool = False) -> RAGRunner:
    global _rag_runner
    if _rag_runner is not None and not force_rebuild:
        return _rag_runner

    # ensure vectorstore exists
    ensure_vectorstore()
    _rag_runner = RAGRunner()
    logger.info("RAG runner built (Qdrant + optional KG)")
    return _rag_runner


# ========== Simple CLI test harness ==========
if __name__ == "__main__":
    print("="*60)
    print("RAG with Qdrant - test harness")
    print("="*60)

    # Quick initialization
    try:
        client, vs = init_qdrant_and_vectorstore()
        print("Qdrant connected and vectorstore initialized.")
    except Exception as e:
        print(f"Failed to initialize Qdrant: {e}")

    # Example ingestion (user should replace with real file paths)
    example_txt = "example.txt"
    if Path(example_txt).exists():
        print("Ingesting example.txt ...")
        ingest_text_file(example_txt, {"id": _gen_id("ex"), "filename": example_txt})
        print("Ingest finished.")
    else:
        print("No example.txt found. Create one to test ingestion.")

    # Build RAG runner
    runner = build_rag_runner()

    # Smoke query (edit as needed)
    q = "What is the content of the example file?"
    try:
        res = runner.run(q, top_k=3)
        print("ANSWER:")
        print(res["answer"])
        print("\nSOURCES:")
        for s in res["source_documents"]:
            print(s)
    except Exception as e:
        print(f"RAG run failed: {e}")
