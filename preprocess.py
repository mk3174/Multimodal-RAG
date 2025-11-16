import os
import logging
from whisper import load_model
from PIL import Image
import pytesseract
from transformers import BlipProcessor, BlipForConditionalGeneration


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    blip_processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        use_fast=True
    )
    blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    logger.info("BLIP model loaded successfully")
except Exception as e:
    logger.warning(f"Failed to load BLIP model: {e}")
    blip_processor = None
    blip_model = None

whisper_model = None
try:
    pass
    whisper_model = load_model("small")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.warning(f"Failed to load Whisper model: {e}")
    whisper_model = None


def ocr_image(path: str) -> str:
    try:
        img = Image.open(path).convert("RGB")
        text = pytesseract.image_to_string(img)
        logger.info(f"OCR completed for {path}")
        return text.strip()
    except Exception as e:
        logger.error(f"OCR failed for {path}: {e}")
        return ""


def caption_image(path: str) -> str:
    if not blip_processor or not blip_model:
        logger.warning("BLIP model not available, skipping captioning")
        return ""

    try:
        img = Image.open(path).convert("RGB")
        inputs = blip_processor(img, return_tensors="pt")
        outputs = blip_model.generate(**inputs)
        caption = blip_processor.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Caption generated for {path}: {caption}")
        return caption.strip()
    except Exception as e:
        logger.error(f"Image captioning failed for {path}: {e}")
        return ""


def transcribe_audio(path: str):
    if not whisper_model:
        logger.warning("Whisper model not available, skipping transcription")
        return "", []

    try:
        result = whisper_model.transcribe(path)
        text = result.get("text", "").strip()
        segments = result.get("segments", [])
        logger.info(f"Audio transcribed for {path}")
        return text, segments
    except Exception as e:
        logger.error(f"Audio transcription failed for {path}: {e}")
        return "", []
