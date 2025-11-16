import io
import os
import tempfile
import uuid
import logging
import fitz  
from pathlib import Path
from typing import List, Dict, Any, Tuple

from PIL import Image
from langchain_core.documents import Document

# local preprocess helpers (ocr_image, caption_image, transcribe_audio)
from preprocess import ocr_image, caption_image, transcribe_audio

# vectorstore and kg client (may be None)
from langchain_rag import vectorstore
from kg_builder import kg_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> List[str]:
    """Split text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def extract_text_and_images(path: str) -> Tuple[str, List[Image.Image]]:
    """
    Extract text and images from a PDF file using PyMuPDF.
    Returns combined text (joined per-page) and list of PIL.Image objects.
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not installed; cannot extract from PDF.")
    try:
        doc = fitz.open(path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF {path}: {e}")

    all_text_parts = []
    images: List[Image.Image] = []

    try:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            # extract text
            page_text = page.get_text("text") or ""
            if page_text.strip():
                all_text_parts.append(page_text.strip())

            # extract images on page
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image.get("image")
                    if image_bytes:
                        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                        images.append(pil_img)
                except Exception as ie:
                    logger.warning(f"Failed to extract image on page {page_num} index {img_index}: {ie}")
                    continue
    finally:
        doc.close()

    combined_text = "\n\n".join(all_text_parts)
    logger.info(f"PDF extraction finished. chars_text={len(combined_text)}, images_extracted={len(images)}")
    return combined_text, images


def _save_pil_image_to_tempfile(img: Image.Image, suffix: str = ".png") -> str:
    """Save PIL image to a temporary file and return the file path."""
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        img.save(tf, format="PNG")
        tf.flush()
        tf.close()
        return tf.name
    except Exception:
        try:
            tf.close()
            os.unlink(tf.name)
        except Exception:
            pass
        raise


def ingest_text_file(path: str, metadata: dict) -> Dict[str, Any]:
    """
    Ingest a text or PDF file into the vectorstore and knowledge graph.
    For PDFs: extract both page text and embedded images (run OCR + caption on images).
    Image-derived text is indexed as 'image' modality documents referencing page/image index.
    """
    ext = Path(path).suffix.lower()
    source_id = metadata.get("source_id") or str(uuid.uuid4())
    metadata["source_id"] = source_id
    metadata["filename"] = metadata.get("filename") or Path(path).name

    # Extract text and images if PDF, else read text file
    if ext == ".pdf":
        try:
            text, images = extract_text_and_images(path)
        except RuntimeError as e:
            return {"status": "error", "source_id": source_id, "message": str(e), "chunks": 0}
    else:
        # Non-pdf text file
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
            images = []
        except Exception as e:
            logger.error(f"Failed to read text file {path}: {e}")
            return {"status": "error", "source_id": source_id, "message": f"Failed to read file: {e}", "chunks": 0}

    docs_to_add: List[Document] = []
    total_chunks = 0

    # Process page text (if any)
    if text and text.strip():
        text_chunks = chunk_text(text)
        total_chunks += len(text_chunks)
        for i, chunk in enumerate(text_chunks):
            md = {
                **metadata,
                "chunk": i,
                "source_id": source_id,
                "modality": "text",
                "file_type": ext,
                "total_chunks": len(text_chunks),
            }
            docs_to_add.append(Document(page_content=chunk, metadata=md))
        logger.info(f"Created {len(text_chunks)} text chunks from {path}")
    else:
        logger.info(f"No page text extracted from {path}")

    # Process images extracted from PDF (run OCR + caption)
    tmp_files_to_cleanup: List[str] = []
    try:
        for img_idx, pil_img in enumerate(images):
            # save temporary image file so your existing ocr_image/caption_image (which expect paths) can operate
            try:
                tmp_path = _save_pil_image_to_tempfile(pil_img)
                tmp_files_to_cleanup.append(tmp_path)
            except Exception as e:
                logger.warning(f"Could not save extracted image {img_idx} to tempfile: {e}")
                continue

            # use your existing preprocess functions
            try:
                ocr_text = ocr_image(tmp_path)  # expects path
            except Exception as e:
                logger.warning(f"OCR failed for extracted image {img_idx}: {e}")
                ocr_text = ""

            try:
                caption = caption_image(tmp_path)
            except Exception as e:
                logger.warning(f"Captioning failed for extracted image {img_idx}: {e}")
                caption = ""

            combined_parts = []
            if caption:
                combined_parts.append(f"Caption: {caption}")
            if ocr_text:
                combined_parts.append(f"OCR Text: {ocr_text}")

            if combined_parts:
                combined = "\n\n".join(combined_parts)
            else:
                # fallback textual placeholder if no text was derivable
                combined = f"[Embedded image in PDF: {metadata.get('filename')} | image_index={img_idx}]"

            image_chunks = chunk_text(combined)
            total_chunks += len(image_chunks)

            for j, chunk in enumerate(image_chunks):
                md = {
                    **metadata,
                    "chunk": j,
                    "source_id": source_id,
                    "modality": "image",
                    "image_index": img_idx,
                    "has_caption": bool(caption),
                    "has_ocr": bool(ocr_text),
                    "total_chunks": len(image_chunks),
                }
                docs_to_add.append(Document(page_content=chunk, metadata=md))

            logger.info(f"Created {len(image_chunks)} chunks from embedded image {img_idx}")
    finally:
        # cleanup temporary image files
        for p in tmp_files_to_cleanup:
            try:
                os.unlink(p)
            except Exception:
                pass

    if not docs_to_add:
        logger.warning(f"No data extracted from {path}")
        return {"status": "error", "source_id": source_id, "message": "No content extracted", "chunks": 0}

    # Add documents to vectorstore
    if vectorstore is not None:
        try:
            vectorstore.add_documents(docs_to_add)
            logger.info(f"Added {len(docs_to_add)} documents to vectorstore (text+images)")
        except Exception as e:
            logger.error(f"Failed to add documents to vectorstore: {e}")
            return {"status": "error", "source_id": source_id, "chunks": total_chunks, "message": f"Vectorstore failed: {e}"}
    else:
        logger.warning("Vectorstore unavailable; skipped indexing.")

    # Add to knowledge graph
    if kg_client is not None:
        try:
            kg_client.add_document(
                source_id,
                {
                    "filename": metadata.get("filename"),
                    "type": "pdf" if ext == ".pdf" else "text",
                    "file_type": ext,
                    "num_chunks": total_chunks,
                    "num_images": len(images),
                },
            )
            logger.info(f"Added document entry to KG: {source_id}")
        except Exception as e:
            logger.error(f"Failed to add document to KG: {e}")
            return {"status": "warning", "source_id": source_id, "chunks": total_chunks, "message": f"KG failed: {e}"}
    else:
        logger.warning("KG client unavailable; skipped KG indexing.")

    return {"status": "ok", "source_id": source_id, "chunks": total_chunks}


def ingest_image_file(path: str, metadata: dict) -> Dict[str, Any]:
    """
    Ingest an image file (runs OCR + captioning) into vectorstore and KG.
    Kept for compatibility with direct image uploads.
    """
    source_id = metadata.get("source_id") or str(uuid.uuid4())
    metadata["source_id"] = source_id
    metadata["filename"] = metadata.get("filename") or Path(path).name

    logger.info(f"Processing standalone image: {path}")
    try:
        ocr_text = ocr_image(path)
    except Exception as e:
        logger.warning(f"OCR failed for image {path}: {e}")
        ocr_text = ""

    try:
        caption = caption_image(path)
    except Exception as e:
        logger.warning(f"Captioning failed for image {path}: {e}")
        caption = ""

    parts = []
    if caption:
        parts.append(f"Caption: {caption}")
    if ocr_text:
        parts.append(f"OCR Text: {ocr_text}")

    combined = "\n\n".join(parts) if parts else f"[Image file: {metadata.get('filename')}]"
    chunks = chunk_text(combined)

    docs = []
    for i, chunk in enumerate(chunks):
        md = {
            **metadata,
            "chunk": i,
            "source_id": source_id,
            "modality": "image",
            "ocr_text": ocr_text,
            "caption": caption,
            "total_chunks": len(chunks),
        }
        docs.append(Document(page_content=chunk, metadata=md))

    if vectorstore is not None:
        try:
            vectorstore.add_documents(docs)
            logger.info(f"Added {len(docs)} image chunks to vectorstore")
        except Exception as e:
            logger.error(f"Failed to add image docs to vectorstore: {e}")
            return {"status": "error", "source_id": source_id, "chunks": len(chunks), "message": f"Vectorstore failed: {e}"}
    else:
        logger.warning("Vectorstore unavailable; skipped indexing.")

    if kg_client is not None:
        try:
            kg_client.add_document(
                source_id,
                {
                    "filename": metadata.get("filename"),
                    "type": "image",
                    "has_caption": bool(caption),
                    "has_ocr": bool(ocr_text),
                    "num_chunks": len(chunks),
                },
            )
            logger.info(f"Added image document to KG: {source_id}")
        except Exception as e:
            logger.error(f"Failed to add image to KG: {e}")
            return {"status": "warning", "source_id": source_id, "chunks": len(chunks), "message": f"KG failed: {e}"}
    else:
        logger.warning("KG client unavailable; skipped KG indexing.")

    return {"status": "ok", "source_id": source_id, "chunks": len(chunks)}


def ingest_audio_file(path: str, metadata: dict) -> Dict[str, Any]:
    """Ingest an audio file (transcription) into vectorstore and KG."""
    source_id = metadata.get("source_id") or str(uuid.uuid4())
    metadata["source_id"] = source_id
    metadata["filename"] = metadata.get("filename") or Path(path).name

    logger.info(f"Transcribing audio: {path}")
    try:
        text, segments = transcribe_audio(path)
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return {"status": "error", "source_id": source_id, "chunks": 0, "message": f"Transcription failed: {e}"}

    if not text.strip():
        text = f"[Audio file: {metadata.get('filename', 'unknown')}]"

    chunks = chunk_text(text)
    docs = []
    for i, chunk in enumerate(chunks):
        md = {
            **metadata,
            "chunk": i,
            "source_id": source_id,
            "modality": "audio",
            "timestamps": segments,
            "total_chunks": len(chunks),
        }
        docs.append(Document(page_content=chunk, metadata=md))

    if vectorstore is not None:
        try:
            vectorstore.add_documents(docs)
            logger.info(f"Added {len(docs)} audio chunks to vectorstore")
        except Exception as e:
            logger.error(f"Failed to add audio docs to vectorstore: {e}")
            return {"status": "error", "source_id": source_id, "chunks": len(chunks), "message": f"Vectorstore failed: {e}"}
    else:
        logger.warning("Vectorstore unavailable; skipped indexing.")

    if kg_client is not None:
        try:
            kg_client.add_document(
                source_id,
                {
                    "filename": metadata.get("filename"),
                    "type": "audio",
                    "duration_segments": len(segments),
                    "num_chunks": len(chunks),
                },
            )
            logger.info(f"Added audio document to KG: {source_id}")
        except Exception as e:
            logger.error(f"Failed to add audio to KG: {e}")
            return {"status": "warning", "source_id": source_id, "chunks": len(chunks), "message": f"KG failed: {e}"}
    else:
        logger.warning("KG client unavailable; skipped KG indexing.")

    return {"status": "ok", "source_id": source_id, "chunks": len(chunks)}
