import streamlit as st
import uuid
import os
import logging
from pathlib import Path
from typing import Optional

# Import local functions
from main import ingest_text, ingest_image, ingest_audio, query_rag, get_system_status
import kg_builder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Multimodal Graph-RAG",
    page_icon="📚",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📚 Multimodal Graph-RAG System</div>', unsafe_allow_html=True)
st.markdown("**Ingest and query across Text, Images, and Audio with Vector Search + Knowledge Graph**")

# Initialize session state
if "source_map" not in st.session_state:
    st.session_state["source_map"] = {}  # source_id -> local path mapping
if "ingest_history" not in st.session_state:
    st.session_state["ingest_history"] = []

# Sidebar - System Status
with st.sidebar:
    st.header(" System Status")
    
    if st.button("Check Status"):
        with st.spinner("Checking system status..."):
            try:
                status = get_system_status()
                st.success("✓ System Online")
                st.json(status)
            except Exception as e:
                st.error(f"Status check failed: {e}")
    
    st.markdown("---")
    st.markdown("###  Session Stats")
    st.metric("Files Ingested", len(st.session_state["ingest_history"]))
    st.metric("Sources Tracked", len(st.session_state["source_map"]))

# ========== Ingest Section ==========
st.markdown('<div class="section-header"> Ingest Files</div>', unsafe_allow_html=True)

with st.expander("📁 Upload files to index (text / image / audio)", expanded=True):
    st.markdown("""
    *Supported formats:*
    - Text: `.txt`, `.pdf`
    - Images: `.png`, `.jpg`, `.jpeg`
    - Audio: `.mp3`, `.wav`
    """)
    
    uploaded = st.file_uploader(
        "Select one or more files",
        accept_multiple_files=True,
        type=["txt", "pdf", "png", "jpg", "jpeg", "mp3", "wav"]
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ingest_button = st.button("🚀 Ingest Files", type="primary", width="stretch")
    with col2:
        if uploaded:
            st.info(f"📦 {len(uploaded)} file(s) selected")
    
    if ingest_button:
        if not uploaded:
            st.warning("⚠️ Please upload at least one file.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            ingest_results = []
            
            for idx, f in enumerate(uploaded):
                status_text.text(f"Processing {idx + 1}/{len(uploaded)}: {f.name}")
                
                # Create temp file
                tmp_dir = Path("/tmp/rag_uploads")
                tmp_dir.mkdir(exist_ok=True)
                tmp_name = tmp_dir / f"{uuid.uuid4()}_{f.name}"
                
                try:
                    # Write uploaded file to disk
                    with open(tmp_name, "wb") as wf:
                        wf.write(f.getvalue())
                    
                    # Determine file type and ingest
                    lower = f.name.lower()
                    if lower.endswith((".png", ".jpg", ".jpeg")):
                        res = ingest_image(str(tmp_name), filename=f.name)
                    elif lower.endswith((".mp3", ".wav")):
                        res = ingest_audio(str(tmp_name), filename=f.name)
                    else:
                        res = ingest_text(str(tmp_name), filename=f.name)
                    
                    source_id = res.get("source_id")
                    ingest_results.append({
                        "name": f.name,
                        "source_id": source_id,
                        "status": res.get("status", "ok"),
                        "chunks": res.get("chunks", "N/A")
                    })
                    
                    # Store mapping for preview
                    if source_id:
                        st.session_state["source_map"][source_id] = {
                            "path": str(tmp_name),
                            "filename": f.name
                        }
                        st.session_state["ingest_history"].append({
                            "filename": f.name,
                            "source_id": source_id
                        })
                    
                except Exception as e:
                    logger.error(f"Ingest failed for {f.name}: {e}")
                    ingest_results.append({
                        "name": f.name,
                        "status": "error",
                        "error": str(e)
                    })
                
                progress_bar.progress((idx + 1) / len(uploaded))
            
            status_text.empty()
            progress_bar.empty()
            
            # Show results
            st.success(f"✅ Ingestion complete! Processed {len(uploaded)} file(s)")
            
            # Display results table
            results_display = []
            for r in ingest_results:
                results_display.append({
                    "File": r["name"],
                    "Status": "✓" if r["status"] == "ok" else "✗",
                    "Chunks": r.get("chunks", "N/A"),
                    "Source ID": r.get("source_id", "N/A")[:8] + "..." if r.get("source_id") else "N/A"
                })
            
            st.dataframe(results_display, width="stretch")

# ========== Query / RAG Section ==========
st.markdown('<div class="section-header">🔍 Query System</div>', unsafe_allow_html=True)

query = st.text_input(
    "" \
    " Ask a question about your ingested documents",
    placeholder="e.g., What are the main topics discussed in the documents?"
)

col1, col2, col3 = st.columns([2, 1, 3])
with col2:
    query_button = st.button(" Search", type="primary", width="stretch")

if query_button:
    if not query:
        st.warning(" Please enter a question.")
    else:
        with st.spinner(" Searching and generating answer..."):
            try:
                result = query_rag(query)
                
                # Display answer
                st.markdown("### Answer")
                st.markdown(f"> {result.get('answer', '—')}")
                
                # Display sources
                st.markdown("### Source Documents")
                sources = result.get("source_documents", [])
                
                if not sources:
                    st.info("No source documents found.")
                else:
                    for i, src in enumerate(sources):
                        md = src.get("metadata", {}) or {}
                        content = src.get("content", "") or ""
                        src_id = md.get("source_id", f"src_{i}")
                        
                        with st.expander(
                            f"📄 [{i}] {md.get('filename', 'Unknown')} - {md.get('modality', 'text')}",
                            expanded=(i == 0)
                        ):
                            # Metadata info
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.markdown(f"**Source ID:** `{src_id[:16]}...`")
                            with col2:
                                st.markdown(f"**Modality:** {md.get('modality', 'unknown')}")
                            with col3:
                                chunk_num = md.get('chunk')
                                total_chunks = md.get('total_chunks', '?')
                                if chunk_num is not None:
                                    st.markdown(f"**Chunk:** {chunk_num + 1}/{total_chunks}")
                            
                            # Content preview
                            st.markdown("**Content:**")
                            st.code(content[:1000] + ("..." if len(content) > 1000 else ""), language="text")
                            
                            # File preview if available
                            mapping = st.session_state["source_map"].get(src_id)
                            if not mapping and md.get("filename"):
                                # Try to find by filename
                                for k, v in st.session_state["source_map"].items():
                                    if v.get("filename") == md.get("filename"):
                                        mapping = v
                                        break
                            
                            if mapping and os.path.exists(mapping.get("path", "")):
                                local_path = mapping.get("path")
                                st.markdown("**Preview:**")
                                
                                if md.get("modality") == "image" or local_path.lower().endswith((".png", ".jpg", ".jpeg")):
                                    st.image(local_path, caption=md.get("filename"), width="stretch")
                                elif md.get("modality") == "audio" or local_path.lower().endswith((".mp3", ".wav")):
                                    st.audio(local_path)
                                elif local_path.lower().endswith(".pdf"):
                                    st.info(" PDF file - download to view full content")
                                else:
                                    try:
                                        preview_text = Path(local_path).read_text(encoding="utf-8")
                                        st.text_area("File content", preview_text[:2000], height=150)
                                    except Exception:
                                        st.info("Preview not available")
            
            except Exception as e:
                st.error(f"❌ Query failed: {e}")
                logger.error(f"Query error: {e}", exc_info=True)

# ========== Knowledge Graph Exploration ==========
st.markdown('<div class="section-header">🕸️ Knowledge Graph Explorer</div>', unsafe_allow_html=True)

with st.expander("🔍 Search entities in knowledge graph"):
    entity_query = st.text_input(
        "Search for entities (substring match)",
        key="kg_find",
        placeholder="e.g., Python, machine learning, etc."
    )

    if st.button("Search KG", width="stretch"):
        if not entity_query:
            st.warning("Enter an entity search string.")
        elif not kg_builder.kg_client:
            st.error("Knowledge Graph is not available. Check Neo4j configuration.")
        else:
            try:
                with st.spinner("Searching knowledge graph..."):
                    names = kg_builder.kg_client.find_entities_by_text(entity_query)
                
                if not names:
                    st.info(f"No entities found matching '{entity_query}'")
                else:
                    st.success(f"Found {len(names)} matching entities")
                    
                    # Display entities
                    st.markdown("**Matching Entities:**")
                    for name in names:
                        st.markdown(f"- {name}")
                    
                    # Show neighbors for first entity
                    if names:
                        first_entity = names[0]
                        st.markdown(f"** Connected documents for '{first_entity}':**")
                        
                        neighbors = kg_builder.kg_client.get_neighbor_documents(first_entity, hops=1)
                        
                        if neighbors:
                            for doc in neighbors[:5]:  # Show first 5
                                md = doc.metadata
                                snippet = doc.page_content[:200]
                                st.markdown(f"- **{md.get('filename', 'Unknown')}** ({md.get('type', 'unknown')})")
                                st.caption(f"  {snippet}...")
                        else:
                            st.info("No connected documents found.")
            
            except Exception as e:
                st.error(f"KG query failed: {e}")
                logger.error(f"KG error: {e}", exc_info=True)

# ========== Footer ==========
st.markdown("---")
st.caption("""
**Notes:**
- OCR and audio transcription may be slow depending on file size and hardware
- Uploaded files are stored temporarily and will be cleared when the session ends
- For production use, consider implementing persistent file storage
- The system uses hybrid retrieval: vector similarity search + knowledge graph expansion
""")

# Debug info in sidebar
with st.sidebar:
    st.markdown("---")
    if st.checkbox("Show Debug Info"):
        st.markdown("### Debug Information")
        st.json({
            "session_id": id(st.session_state),
            "sources_tracked": len(st.session_state["source_map"]),
            "history_length": len(st.session_state["ingest_history"])
        })