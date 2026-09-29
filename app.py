"""
FastAPI application for Form-C document information extraction.
Integrates synchronous Qwen2-VL vision endpoints with per-field confidence scoring,
Tesseract MRZ parsing, and an asynchronous sequential batch processing queue for multi-document workflows.
"""

import asyncio
from contextlib import asynccontextmanager
import io
import json
import logging
import os
import re
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Configuration and modular routers
from config import settings
from routers.mrz_router import router as mrz_router
from routers.jobs_router import router as jobs_router

# Database persistence and async background queue worker
from storage.db import initialize_database, reset_interrupted_jobs
from services.job_worker import background_queue_worker, job_queue

# Schema models
from schemas.job import FieldResult

# Configure application-level logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("passport_api")

# Global dictionary holding loaded model artifacts in system memory
ml_models: Dict[str, Any] = {}


# ==============================================================================
# Synchronous Response Schema for Passport Bio
# ==============================================================================

class PassportBioExtractionResponse(BaseModel):
    """
    Synchronous response structure for Qwen2-VL passport extraction returning
    per-field confidence metrics and source attribution.
    """
    # Extracted surname
    surname: FieldResult = Field(default_factory=FieldResult, description="Holder's surname / family name")

    # Extracted given names
    given_names: FieldResult = Field(default_factory=FieldResult, description="Holder's given names")

    # Document identification number
    passport_number: FieldResult = Field(default_factory=FieldResult, description="Passport identifier number")

    # 3-letter ICAO country code or country name
    nationality: FieldResult = Field(default_factory=FieldResult, description="Holder's nationality")

    # Birth date (YYYY-MM-DD)
    date_of_birth: FieldResult = Field(default_factory=FieldResult, description="Date of birth")

    # Holder's sex ('M', 'F', 'X')
    sex: FieldResult = Field(default_factory=FieldResult, description="Sex code")

    # Passport expiry date (YYYY-MM-DD)
    passport_expiry_date: FieldResult = Field(default_factory=FieldResult, description="Passport expiration date")

    # Diagnostic observation from the vision model
    confidence_note: Optional[str] = Field(default=None, description="Model explanation of image legibility")

    # Status indicator string
    status: str = Field(default="success", description="Execution result status")


# ==============================================================================
# Application Lifecycle Management (Lifespan)
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events:
    1. Loads Qwen2-VL model and processor into RAM.
    2. Configures PyTorch compute thread pinning for CPU.
    3. Initializes the SQLite jobs database table.
    4. Recovers interrupted jobs after system reboot.
    5. Starts the background queue worker task.
    6. Cancels background tasks on shutdown.
    """
    logger.info(f"[*] Initializing model '{settings.model_id}' on device: {settings.device.upper()}...")

    # PyTorch CPU thread pinning using safe getattr
    torch_threads = getattr(settings, "torch_cpu_threads", None)
    if settings.device.lower() == "cpu" and torch_threads is not None and torch_threads > 0:
        torch.set_num_threads(torch_threads)
        logger.info(f"[*] PyTorch CPU compute threads configured to: {torch_threads}")
    else:
        logger.info("[*] PyTorch CPU compute threads using system default allocation.")

    compute_dtype = torch.bfloat16 if settings.device.lower() == "cpu" else torch.float16

    try:
        # Load processor with bounded visual token limits
        processor = AutoProcessor.from_pretrained(
            settings.model_id,
            min_pixels=settings.min_pixels,
            max_pixels=settings.max_pixels
        )

        # Load Qwen2-VL weights in evaluation mode
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            settings.model_id,
            torch_dtype=compute_dtype,
            device_map=settings.device
        )
        model.eval()

        ml_models["model"] = model
        ml_models["processor"] = processor
        logger.info("[*] Model and processor successfully loaded and ready for queries.")

    except Exception as exc:
        logger.error(f"[!] Critical failure loading model '{settings.model_id}': {exc}")
        raise exc

    # 1. Initialize SQLite jobs table
    initialize_database()

    # 2. Crash recovery: re-queue any jobs interrupted mid-processing during power loss
    interrupted_job_ids = reset_interrupted_jobs()
    if interrupted_job_ids:
        logger.info(f"[*] Recovered {len(interrupted_job_ids)} interrupted job(s). Re-queuing...")
        for j_id in interrupted_job_ids:
            await job_queue.put(j_id)

    # 3. Start background queue worker as a persistent asyncio background task
    worker_task = asyncio.create_task(background_queue_worker(ml_models))
    logger.info("[*] Background queue worker task started.")

    yield

    # Clean shutdown: cancel the queue worker
    logger.info("[*] Shutting down application. Cancelling background queue worker...")
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass

    # Clear model artifacts from RAM
    ml_models.clear()
    logger.info("[*] Application shutdown complete.")


# ==============================================================================
# FastAPI Application Instantiation
# ==============================================================================

app_title = getattr(settings, "api_title", "Passport and Visa OCR Extraction API")
app_version = getattr(settings, "api_version", "1.0.0")
app_description = getattr(
    settings,
    "api_description",
    "High-throughput document extraction combining Qwen2-VL vision models, Tesseract MRZ parsing, and asynchronous batch processing."
)

app = FastAPI(
    title=app_title,
    version=app_version,
    description=app_description,
    lifespan=lifespan
)

# CORS middleware for mobile and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount modular routers
app.include_router(mrz_router)
app.include_router(jobs_router, prefix="/api/v1")


# ==============================================================================
# Synchronous Helper Functions
# ==============================================================================

def validate_and_load_image(file_bytes: bytes) -> Image.Image:
    """
    Validates byte stream and loads as RGB PIL Image.
    """
    try:
        image = Image.open(io.BytesIO(file_bytes))
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupt image file: {str(exc)}"
        )


def execute_vlm_query(image: Image.Image, system_prompt: str) -> dict:
    """
    Runs inference through the warm Qwen2-VL model and returns parsed JSON.
    """
    processor = ml_models.get("processor")
    model = ml_models.get("model")

    if not model or not processor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not yet initialized or ready."
        )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": settings.min_pixels,
                    "max_pixels": settings.max_pixels
                },
                {"type": "text", "text": system_prompt}
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(settings.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            do_sample=False
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    raw_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(raw_text)
    except Exception:
        return {"raw_output": raw_text, "confidence_note": "Failed to parse structured JSON."}


def build_field_result_from_vlm(raw_value: Optional[str], raw_conf: Any) -> FieldResult:
    """
    Normalizes VLM extraction and confidence into a validated FieldResult.
    Applies heuristic validation to bound confidence between 0.0 and 1.0.
    """
    if not raw_value or str(raw_value).strip().lower() in ["null", "none", ""]:
        return FieldResult(value=None, confidence=0.0, source="qwen2_vl")

    cleaned_val = str(raw_value).strip()

    # Parse confidence float
    try:
        score = float(raw_conf)
        score = max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        score = 0.70  # Default fallback score for detected text without explicit valid float

    return FieldResult(
        value=cleaned_val,
        confidence=round(score, 2),
        source="qwen2_vl"
    )


# ==============================================================================
# Synchronous VLM Endpoints
# ==============================================================================

@app.post(
    "/api/v1/extract/passport",
    response_model=PassportBioExtractionResponse,
    summary="Synchronous Passport Bio-Data Extraction via Qwen2-VL with Field Confidence",
    tags=["Synchronous Extraction"]
)
async def extract_passport_bio(file: UploadFile = File(...)) -> PassportBioExtractionResponse:
    """
    Extracts passport bio-data fields along with individual visual confidence scores
    rated from 0.0 (uncertain/blurred) to 1.0 (crystal clear).
    """
    image_bytes = await file.read()
    image = validate_and_load_image(image_bytes)

    prompt = (
        "Carefully read the passport bio page image. For each field, provide the extracted string value "
        "and your visual confidence score between 0.0 and 1.0 based on clarity and legibility. "
        "Return valid JSON formatted exactly with this structure:\n"
        "{\n"
        '  "surname": {"value": "...", "confidence": 0.95},\n'
        '  "given_names": {"value": "...", "confidence": 0.95},\n'
        '  "passport_number": {"value": "...", "confidence": 0.90},\n'
        '  "nationality": {"value": "...", "confidence": 0.95},\n'
        '  "date_of_birth": {"value": "YYYY-MM-DD", "confidence": 0.90},\n'
        '  "sex": {"value": "M or F", "confidence": 0.95},\n'
        '  "expiry_date": {"value": "YYYY-MM-DD", "confidence": 0.90},\n'
        '  "confidence_note": "short explanation of image quality"\n'
        "}"
    )

    raw_response = execute_vlm_query(image, prompt)

    # Helper to parse nested or flat response patterns from VLM
    def parse_field(field_name: str) -> FieldResult:
        entry = raw_response.get(field_name)
        if isinstance(entry, dict):
            return build_field_result_from_vlm(entry.get("value"), entry.get("confidence", 0.85))
        return build_field_result_from_vlm(entry, raw_response.get(f"{field_name}_confidence", 0.80))

    return PassportBioExtractionResponse(
        surname=parse_field("surname"),
        given_names=parse_field("given_names"),
        passport_number=parse_field("passport_number"),
        nationality=parse_field("nationality"),
        date_of_birth=parse_field("date_of_birth"),
        sex=parse_field("sex"),
        passport_expiry_date=parse_field("expiry_date"),
        confidence_note=raw_response.get("confidence_note", "Visual extraction complete."),
        status="success"
    )


@app.post(
    "/api/v1/extract/visa",
    summary="Synchronous Visa / OCI Extraction via Qwen2-VL",
    tags=["Synchronous Extraction"]
)
async def extract_visa_details(file: UploadFile = File(...)):
    """
    Extracts visa number, type, issue date, expiry date, and place of issue.
    """
    image_bytes = await file.read()
    image = validate_and_load_image(image_bytes)
    prompt = (
        "Extract visa details from the image. Output valid JSON with keys: "
        "'visa_number', 'visa_type', 'issue_date', 'expiry_date', 'place_of_issue', 'confidence_note'."
    )
    return execute_vlm_query(image, prompt)


@app.post(
    "/api/v1/extract/stamp",
    summary="Synchronous Immigration Stamp Extraction via Qwen2-VL",
    tags=["Synchronous Extraction"]
)
async def extract_stamp_details(file: UploadFile = File(...)):
    """
    Extracts arrival date and port of entry from an Indian immigration ink stamp.
    """
    image_bytes = await file.read()
    image = validate_and_load_image(image_bytes)
    prompt = (
        "Extract arrival stamp details from the image. Output valid JSON with keys: "
        "'arrival_date', 'arrival_port', 'confidence_note'."
    )
    return execute_vlm_query(image, prompt)


# ==============================================================================
# Diagnostic Probes
# ==============================================================================

@app.get("/health", tags=["Health"])
def health_check():
    """
    Sanitized root diagnostic probe indicating operational readiness of the API
    and warm status of the vision model.
    """
    is_ready = "model" in ml_models and "processor" in ml_models
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
        "device": settings.device,
        "max_pixels": settings.max_pixels,
        "min_pixels": settings.min_pixels
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)