import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from openai import OpenAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()

QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_COLLECTION = os.getenv('QDRANT_COLLECTION')
EMBED_MODEL = os.getenv('EMBEDDING_MODEL')
LLM_MODEL = os.getenv('LLM_MODEL')
OPENAI_API = os.getenv('OPENAI_API_KEY')

embeddings = OpenAIEmbeddings(model=EMBED_MODEL,dimensions=1536)


vectorstore = None
client = None
try:
    client = QdrantClient(url=QDRANT_URL,api_key=QDRANT_API_KEY)
    
    # Optional: Check if the collection exists (basic check, can be verbose)
    collections = client.get_collections().collections
    if not any(c.name == QDRANT_COLLECTION for c in collections):
        logger.warning(f"Qdrant collection '{QDRANT_COLLECTION}' not found. Make sure it is created and indexed.")

    # 2. Initialize LangChain QdrantVectorStore
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embeddings,
        # FIX: Specify the vector name used in the Qdrant collection
        vector_name="hybrid search",
        # Set the key where the main text is stored in Qdrant's payload
        content_payload_key="page_content" 
    )
    logger.info(f"Qdrant initialized: {QDRANT_COLLECTION} at {QDRANT_URL}")

except Exception as e:
    logger.warning(f"Failed to initialize Qdrant: {e}")
    logger.warning("Qdrant vectorstore is unavailable. Vector search will be disabled.")
    logger.warning("Please check your Qdrant configuration, ensure it is running, and the collection exists.")
    vectorstore = None
    client = None


llm = ChatOpenAI(api_key=OPENAI_API, model=LLM_MODEL)


class HybridRetriever(BaseRetriever):
    """Retriever that combines vector search with knowledge graph expansion."""
    
    vectorstore: Any
    kg: Any = None
    kg_hops: int = 1

    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        return self.get_relevant_documents(query)
    
    async def _aget_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        return self.get_relevant_documents(query)
    
    def get_relevant_documents(self, query: str, k: int = 6) -> List[Document]:
        """Get relevant documents using vector search + optional KG expansion."""
        logger.info(f"Retrieving documents for query: {query[:100]}...")
        
        # 1) Vector search 
        try:
            # QdrantVectorStore implements similarity_search
            docs = self.vectorstore.similarity_search(query, k=k)
            logger.info(f"Vector search returned {len(docs)} documents")
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            docs = []
        
        # 2) KG expansion
        if self.kg:
            try:
                entities = self.kg.find_entities_by_text(query)
                logger.info(f"Found {len(entities)} entities in query")
                
                for ent in entities:
                    neighbors = self.kg.get_neighbor_documents(ent, hops=self.kg_hops)
                    logger.info(f"Found {len(neighbors)} neighbor documents for entity '{ent}'")
                    docs.extend(neighbors)
            except Exception as e:
                logger.error(f"KG expansion failed: {e}")
        
        # 3) Deduplicate by source_id or content
        seen = set()
        deduped = []
        for d in docs:
            md = getattr(d, 'metadata', {}) or {}
            key = md.get('source_id')
            if not key:
                content = getattr(d, 'page_content', '')
                key = content[:200] if content else id(d)
            
            if key in seen:
                continue
            seen.add(key)
            deduped.append(d)
        
        logger.info(f"Returning {len(deduped)} deduplicated documents")
        return deduped

# --- Helper Function: Format Sources ---

def _format_sources_for_prompt(docs: List[Document]) -> str:
    """Formats retrieved documents into a clean string for the LLM prompt."""
    pieces = []
    for i, d in enumerate(docs):
        md = getattr(d, "metadata", {}) or {}
        snippet = getattr(d, "page_content", "")[:1400].strip()
        src = md.get("source_id") or md.get("filename") or f"source_{i}"
        modality = md.get("modality", "text")
        pieces.append(f"[{i}] source:{src} modality:{modality}\n{snippet}")
    return "\n\n".join(pieces)

# --- Main RAG Chain Builder ---

def build_rag_chain(kg_client=None):
    """Builds and returns the RAG execution object."""
    
    # Ensure vectorstore is available before creating the retriever
    if vectorstore is None:
        logger.error("Cannot build RAG chain: Qdrant vectorstore failed to initialize.")
        return None

    logger.info("Building RAG chain...")
    
    # Create retriever instance
    # Assuming kg_client is used for the kg attribute in HybridRetriever
    retriever = HybridRetriever(vectorstore=vectorstore, kg=kg_client) 
    
    # Define prompt template
    prompt_template = """You are an expert assistant. Use ONLY the provided context to answer the question.
If the answer is not contained in the context, say you don't know.
Be concise and include inline source tags like [0] when referencing specific sources.

CONTEXT:
{context}

QUESTION: {question}

Answer:"""
    
    chat_prompt = ChatPromptTemplate.from_template(prompt_template)
    
    class SimpleRAG:
        """Simple class to execute the RAG pipeline."""
        
        def __init__(self, retriever, llm, prompt):
            self.retriever = retriever
            self.llm = llm
            self.prompt = prompt
        
        def _call_llm_safely(self, messages: List) -> str:
            """Attempts to call the primary LLM, with fallback to raw OpenAI API if needed."""
            
            # 1. Primary LangChain LLM invoke
            try:
                response = self.llm.invoke(messages)
                if hasattr(response, 'content'):
                    return response.content
                return str(response)
            except Exception as e:
                logger.warning(f"Primary LLM invoke failed: {e}, trying fallback...")
            
            # 2. Fallback to raw OpenAI API client
            try:
                # Assuming OPENAI_API_KEY is available in ENV
                client = OpenAI(api_key=os.getenv('OPENAI_API_KEY')) 
                
                # Build messages for OpenAI
                api_messages = []
                for msg in messages:
                    # Convert LangChain Message types to OpenAI dictionary format
                    if hasattr(msg, 'type') and hasattr(msg, 'content'):
                        role = 'assistant' if msg.type == 'ai' else 'user'
                        api_messages.append({"role": role, "content": msg.content})
                    else:
                        api_messages.append({"role": "user", "content": str(msg)})
                
                response = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=api_messages,
                    temperature=0.0,
                    max_tokens=700
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"All LLM call strategies failed: {e}")
                raise RuntimeError(f"LLM call failed: {e}")

        
        def run(self, query: str, top_k: int = 6) -> Dict[str, Any]:
            """Executes the full RAG query process."""
            logger.info(f"Running RAG query: {query[:100]}...")
            
            # 1) Retrieve documents
            try:
                docs = self.retriever.get_relevant_documents(query, k=top_k)
            except Exception as e:
                logger.error(f"Document retrieval failed: {e}")
                docs = []
            
            docs = docs[:top_k]
            
            if not docs:
                logger.warning("No documents retrieved")
                return {
                    "answer": "I don't have enough information to answer this question.",
                    "source_documents": []
                }
            
            # 2) Format context and build prompt
            context = _format_sources_for_prompt(docs)
            
            try:
                prompt_value = self.prompt.format_prompt(context=context, question=query)
                messages = prompt_value.to_messages()
            except Exception as e:
                logger.warning(f"Prompt formatting failed: {e}, using fallback")
                # Fallback to simple string message if to_messages() fails
                messages = [HumanMessage(content=f"{prompt_template.format(context=context, question=query)}")]
            
            try:
                answer = self._call_llm_safely(messages)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                answer = f"Error generating answer: {str(e)}"
            
            # 4) Format output
            source_documents = []
            for d in docs:
                source_documents.append({
                    "content": getattr(d, "page_content", ""),
                    "metadata": getattr(d, "metadata", {}) or {}
                })
            
            logger.info("RAG query completed successfully")
            return {
                "answer": answer,
                "source_documents": source_documents
            }
    
    rag_instance = SimpleRAG(retriever=retriever, llm=llm, prompt=chat_prompt)
    logger.info("RAG chain built successfully")
    return rag_instance