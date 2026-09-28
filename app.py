"""
Passport Stamp, Visa Page & Bio-Data Extraction Service
FastAPI microservice executing deterministic CPU vision inference using Qwen2-VL-2B-Instruct.
Includes optimized token bounds and multithreaded CPU scheduling to prevent Cloudflare 524 timeouts.
"""

import io
import json
import os
import re
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Configuration and schema imports
from config import settings
from schemas.stamp import StampExtractionResponse
from schemas.visa import VisaExtractionResponse
from schemas.passport_bio import PassportBioExtractionResponse


# ==============================================================================
# Model Lifespan Management
# ==============================================================================

# Global state dictionary holding warm VLM model weights and tokenizers
ml_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages startup and shutdown events for the FastAPI application.
    Configures CPU thread scheduling and pre-warms the Qwen2-VL model and processor.
    """
    print(f"[*] Initializing model '{settings.model_id}' on device: {settings.device.upper()}...")

    # Configure PyTorch CPU thread count to maximize execution throughput across all available cores
    available_cores = os.cpu_count() or 4
    active_threads = settings.cpu_threads if settings.cpu_threads > 0 else available_cores
    torch.set_num_threads(active_threads)
    torch.set_num_interop_threads(active_threads)
    print(f"[*] PyTorch CPU compute threads configured to: {active_threads}")

    # AutoProcessor tokenizes image patches and text prompts.
    # min_pixels and max_pixels bound visual token resolution for fast CPU latency.
    processor = AutoProcessor.from_pretrained(
        settings.model_id,
        min_pixels=settings.min_pixels,
        max_pixels=settings.max_pixels
    )

    # Load 32-bit float weights for deterministic execution on CPU hardware
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        settings.model_id,
        torch_dtype=torch.float32,
        device_map=settings.device,
        low_cpu_mem_usage=True
    )

    # Store references in global lifespan dictionary
    ml_state["processor"] = processor
    ml_state["model"] = model
    print("[*] Model and processor successfully loaded and ready for queries.")

    yield

    # Release memory handles when the FastAPI application shuts down
    ml_state.clear()
    print("[*] Model resources freed from memory.")


# ==============================================================================
# Application Definition
# ==============================================================================

app = FastAPI(
    title="Passport Stamp, Visa & Bio-Data Capture API",
    description="Local open-source CPU service for passport entry stamps, visa pages, and visual bio-data.",
    version="1.2.1",
    lifespan=lifespan
)


# ==============================================================================
# Image Processing & Generic Inference Pipeline
# ==============================================================================

def prepare_uploaded_image(upload_file: UploadFile) -> Image.Image:
    """
    Reads incoming multipart binary bytes, confirms image legitimacy, standardizes
    color channels to RGB, and downscales oversized phone photos to maintain CPU latency.
    """
    if upload_file.content_type and not upload_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file MIME type: {upload_file.content_type}. Please upload an image."
        )

    try:
        raw_bytes = upload_file.file.read()
        image = Image.open(io.BytesIO(raw_bytes))

        # Standardize color profile to 3-channel RGB (removes alpha channels or CMYK artifacts)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Downscale proportionally if largest dimension exceeds configured max bounds
        max_dim = settings.max_image_dimension
        if max(image.size) > max_dim:
            image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        return image
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to decode and prepare uploaded image: {str(exc)}"
        )


def execute_vlm_query(image: Image.Image, system_prompt: str) -> dict:
    """
    Executes a structured query against the shared in-memory Qwen2-VL model
    and parses the output into a Python dictionary.
    """
    processor = ml_state.get("processor")
    model = ml_state.get("model")

    if not model or not processor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision model engine is not ready or has not been initialized."
        )

    # Format user prompt according to Qwen2-VL chat conversation template.
    # Explicitly pass min_pixels and max_pixels in the image dict so qwen_vl_utils
    # enforces the token bounds directly on the input image.
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

    # Process prompt template and extract visual input tensors
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    # Build input tensor dictionaries
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    # Run deterministic inference without calculating gradients
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            do_sample=False
        )

    # Separate model-generated response tokens from the input prompt tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode tokens into UTF-8 string
    raw_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Extract JSON string payload bounded by braces
    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(raw_text)
    except Exception:
        return {"confidence_note": f"Raw model response: {raw_text}"}


# ==============================================================================
# Service Endpoints
# ==============================================================================

full_stamp_path = f"{settings.api_v1_prefix}{settings.stamp_extract_path}"
full_visa_path = f"{settings.api_v1_prefix}{settings.visa_extract_path}"
full_bio_path = f"{settings.api_v1_prefix}{settings.passport_bio_extract_path}"


@app.post(
    full_stamp_path,
    response_model=StampExtractionResponse,
    summary="Extract Arrival Date and Port from Immigration Ink Stamp",
    tags=["Immigration Extraction"]
)
async def extract_stamp_data(
    file: UploadFile = File(..., description="Passport page image containing immigration ink stamps")
):
    """
    Locates entry/arrival rubber ink stamps on a passport page and extracts
    arrival date (Form-C Field 5) and port of arrival (Form-C Field 12).
    """
    image = prepare_uploaded_image(file)

    prompt = (
        "You are an immigration stamp inspection assistant.\n"
        "Examine the uploaded image of a passport page containing entry/arrival immigration rubber stamps.\n"
        "Locate the primary ARRIVAL or ENTRY stamp and extract:\n"
        "1. arrival_date: The date stamped on arrival (format strictly as YYYY-MM-DD).\n"
        "2. arrival_port: The port or airport of entry indicated on the stamp (e.g., 'COCHIN SEAPORT', 'DELHI AIRPORT', 'BOMBAY AIRPORT').\n"
        "3. confidence_note: Short note on legibility and stamp identification.\n"
        "Return the output strictly as a JSON object with keys: 'arrival_date', 'arrival_port', 'confidence_note'."
    )

    data = execute_vlm_query(image, prompt)
    return StampExtractionResponse(**data)


@app.post(
    full_visa_path,
    response_model=VisaExtractionResponse,
    summary="Extract Visa Number, Type, and Validity Dates",
    tags=["Immigration Extraction"]
)
async def extract_visa_data(
    file: UploadFile = File(..., description="Image of Indian Visa sticker or Overseas Citizen of India (OCI) page")
):
    """
    Extracts visa number, date of issue, expiry date, and visa category/type.
    """
    image = prepare_uploaded_image(file)

    prompt = (
        "You are an expert immigration document processing assistant.\n"
        "Examine the uploaded visa document or OCI page image and extract:\n"
        "1. visa_number: The unique visa number or sticker number.\n"
        "2. visa_type: The visa category (e.g., 'TOURIST', 'BUSINESS', 'CONFERENCE', 'OCI').\n"
        "3. issue_date: Date of issue (format strictly as YYYY-MM-DD).\n"
        "4. expiry_date: Date of expiry (format strictly as YYYY-MM-DD).\n"
        "5. place_of_issue: Location or authority that issued the visa (e.g., 'LONDON', 'NEW YORK', 'PARIS').\n"
        "6. confidence_note: Brief remarks on clarity or any damaged regions.\n"
        "Return the output strictly as a JSON object with keys: 'visa_number', 'visa_type', 'issue_date', 'expiry_date', 'place_of_issue', 'confidence_note'."
    )

    data = execute_vlm_query(image, prompt)
    return VisaExtractionResponse(**data)


@app.post(
    full_bio_path,
    response_model=PassportBioExtractionResponse,
    summary="Extract Passport Bio Page Details (Visual Fallback)",
    tags=["Immigration Extraction"]
)
async def extract_passport_bio_data(
    file: UploadFile = File(..., description="Passport identity page image")
):
    """
    Extracts visual text fields from passport photo identity page:
    Given names, surname, passport number, nationality, date of birth, sex, and expiry.
    """
    image = prepare_uploaded_image(file)

    prompt = (
        "You are a passport verification assistant.\n"
        "Examine the uploaded passport biographical identity page and extract the following fields:\n"
        "1. surname: The bearer's primary surname / family name.\n"
        "2. given_names: The bearer's first name and any middle names.\n"
        "3. passport_number: The unique document/passport number.\n"
        "4. nationality: The nationality or issuing country.\n"
        "5. date_of_birth: Date of birth (format strictly as YYYY-MM-DD).\n"
        "6. sex: Gender ('M', 'F', or 'X').\n"
        "7. expiry_date: Passport expiration date (format strictly as YYYY-MM-DD).\n"
        "8. confidence_note: Short remarks on document legibility.\n"
        "Return the output strictly as a JSON object with keys: 'surname', 'given_names', 'passport_number', 'nationality', 'date_of_birth', 'sex', 'expiry_date', 'confidence_note'."
    )

    data = execute_vlm_query(image, prompt)
    return PassportBioExtractionResponse(**data)


@app.get("/health", summary="Health check endpoint", tags=["System"])
async def health_check():
    """
    Verifies that the microservice is running and the model is initialized.
    """
    is_ready = ml_state.get("model") is not None and ml_state.get("processor") is not None
    return {
        "status": "healthy" if is_ready else "initializing",
        "model_loaded": is_ready,
        "device": settings.device,
        "max_pixels": settings.max_pixels,
        "min_pixels": settings.min_pixels
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False
    )