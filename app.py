"""
FastAPI application for Form-C document information extraction.
Integrates synchronous Qwen2-VL vision endpoints, Tesseract MRZ parsing,
and an asynchronous sequential batch processing queue for multi-document workflows.
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

# Configure application-level logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("passport_api")

# Global dictionary holding loaded model artifacts in system memory
ml_models: Dict[str, Any] = {}


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

    # Configure PyTorch CPU compute threads for Dell server
    if settings.device.lower() == "cpu" and settings.torch_cpu_threads > 0:
        torch.set_num_threads(settings.torch_cpu_threads)
        logger.info(f"[*] PyTorch CPU compute threads configured to: {settings.torch_cpu_threads}")

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

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
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


# ==============================================================================
# Existing Synchronous VLM Endpoints (Preserved Intact)
# ==============================================================================

@app.post(
    "/api/v1/extract/passport",
    summary="Synchronous Passport Bio-Data Extraction via Qwen2-VL",
    tags=["Synchronous Extraction"]
)
async def extract_passport_bio(file: UploadFile = File(...)):
    """
    Extracts holder surname, given names, passport number, nationality, DOB, and expiry.
    """
    image_bytes = await file.read()
    image = validate_and_load_image(image_bytes)
    prompt = (
        "Extract passport bio data from the image. Output valid JSON with keys: "
        "'surname', 'given_names', 'passport_number', 'nationality', "
        "'date_of_birth', 'sex', 'expiry_date', 'confidence_note'."
    )
    return execute_vlm_query(image, prompt)


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