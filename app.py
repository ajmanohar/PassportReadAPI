"""
app.py
------
FastAPI microservice hosting passport entry stamp extraction,
visa stamp/sticker extraction, and visual bio-data extraction (non-MRZ)
on standard CPU using Qwen2-VL-2B-Instruct.
"""

import io
import json
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

# Dictionary holding in-memory references to the shared processor and model
ml_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads Qwen2-VL-2B-Instruct once into CPU memory upon server boot.
    Shared across visa, stamp, and passport bio extraction endpoints to avoid memory churn.
    """
    print(f"[*] Initializing model '{settings.model_id}' on device: {settings.device.upper()}...")

    # Set PyTorch thread limit to avoid CPU core contention on standard multi-core servers
    # Adjust or leave default based on server hardware threads
    if torch.get_num_threads() > 4:
        torch.set_num_threads(4)

    # AutoProcessor tokenizes image patches and text prompts
    # min_pixels and max_pixels bound visual token resolution for optimal CPU latency
    processor = AutoProcessor.from_pretrained(
        settings.model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=1024 * 28 * 28
    )

    # Load 32-bit float weights for deterministic execution on standard CPU hardware
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
    version="1.2.0",
    lifespan=lifespan
)


# ==============================================================================
# Image Processing & Generic Inference Pipeline
# ==============================================================================

def prepare_uploaded_image(upload_file: UploadFile) -> Image.Image:
    """
    Reads incoming multipart binary bytes, confirms image legitimacy, standardizes
    color channels to RGB, and scales down oversized images to maintain responsive CPU latency.
    """
    # Verify content type starts with image/
    if upload_file.content_type and not upload_file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file MIME type: {upload_file.content_type}. Please upload an image."
        )

    try:
        raw_bytes = upload_file.file.read()
        image = Image.open(io.BytesIO(raw_bytes))

        # Standardize color profile to 3-channel RGB (removes alpha channel or CMYK artifacts)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Downscale proportionally if largest dimension exceeds config bounds
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
    # Fetch active model instances from lifespan storage
    processor = ml_state.get("processor")
    model = ml_state.get("model")

    if not model or not processor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision model engine is not ready or has not been initialized."
        )

    # Format user prompt according to Qwen2-VL chat conversation template
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
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
            max_new_tokens=320,
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

    # Extract JSON string payload bounded by brackets
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

# Dynamic route paths from central settings
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
        "Examine the image for passport immigration ink stamps (rectangular, circular, or oval ink marks).\n"
        "1. Identify the ARRIVAL / ENTRY stamp (often contains 'ARRIVED', 'ENTRY', or incoming arrow).\n"
        "2. Do NOT select EXIT / DEPARTURE stamps.\n"
        "3. Extract the arrival date and normalize it to YYYY-MM-DD.\n"
        "4. Extract the port of arrival (e.g., COCHIN AIRPORT, DEL, BOM, etc.).\n"
        "Return ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "is_stamp_detected": true,\n'
        '  "arrival_date": "YYYY-MM-DD",\n'
        '  "port_of_entry": "string",\n'
        '  "stamp_type": "ENTRY",\n'
        '  "stay_permitted_until": "string or null",\n'
        '  "confidence_note": "string or null"\n'
        "}\n"
        "Do not include any explanation before or after the JSON."
    )

    data = execute_vlm_query(image, prompt)

    return StampExtractionResponse(
        is_stamp_detected=data.get("is_stamp_detected", False),
        arrival_date=data.get("arrival_date"),
        port_of_entry=data.get("port_of_entry"),
        stamp_type=data.get("stamp_type", "ENTRY"),
        stay_permitted_until=data.get("stay_permitted_until"),
        confidence_note=data.get("confidence_note")
    )


@app.post(
    full_visa_path,
    response_model=VisaExtractionResponse,
    summary="Extract Visa Details, Numbers & Endorsements",
    tags=["Immigration Extraction"]
)
async def extract_visa_data(
    file: UploadFile = File(..., description="Passport page image containing a visa sticker or consular stamp")
):
    """
    Extracts visa metadata, including printed/handwritten visa numbers, place of issue
    (Form-C Field 4), issue/expiry dates, and handwritten officer annotations[cite: 1, 2].
    """
    image = prepare_uploaded_image(file)

    prompt = (
        "You are an expert immigration document analyzer.\n"
        "Examine the image of a passport visa page (sticker, e-Visa print, or consular ink stamp).\n"
        "1. Identify if a valid visa is present.\n"
        "2. Extract the Visa Number (alphanumeric, printed or handwritten).\n"
        "3. Extract the Place of Issue (city or consulate name).\n"
        "4. Extract the Date of Issue and Date of Expiry (normalize to YYYY-MM-DD).\n"
        "5. Extract Visa Type and Entries allowed (Single, Double, Multiple).\n"
        "6. Capture any handwritten officer remarks, pen notes, or endorsements.\n"
        "Return ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "is_visa_detected": true,\n'
        '  "visa_number": "string or null",\n'
        '  "place_of_issue": "string or null",\n'
        '  "date_of_issue": "YYYY-MM-DD or null",\n'
        '  "date_of_expiry": "YYYY-MM-DD or null",\n'
        '  "visa_type": "string or null",\n'
        '  "entries_allowed": "string or null",\n'
        '  "handwritten_notes": "string or null",\n'
        '  "confidence_note": "string or null"\n'
        "}\n"
        "Do not include any explanation before or after the JSON."
    )

    data = execute_vlm_query(image, prompt)

    return VisaExtractionResponse(
        is_visa_detected=data.get("is_visa_detected", False),
        visa_number=data.get("visa_number"),
        place_of_issue=data.get("place_of_issue"),
        date_of_issue=data.get("date_of_issue"),
        date_of_expiry=data.get("date_of_expiry"),
        visa_type=data.get("visa_type"),
        entries_allowed=data.get("entries_allowed"),
        handwritten_notes=data.get("handwritten_notes"),
        confidence_note=data.get("confidence_note")
    )


@app.post(
    full_bio_path,
    response_model=PassportBioExtractionResponse,
    summary="Extract Passport Bio-Data from Visual Page (Non-MRZ Fallback)",
    tags=["Immigration Extraction"]
)
async def extract_passport_bio_data(
    file: UploadFile = File(..., description="Passport biodata identity page image (works even if MRZ is cut off or missing)")
):
    """
    Extracts guest identity information directly from the Visual Inspection Zone (VIZ)
    of a passport page when the bottom MRZ lines are obscured, cropped, or illegible.
    Maps directly to Form-C Field 1 (Name) and Field 3 (Nationality)[cite: 7, 8].
    """
    # Preprocess image dimensions and color profile
    image = prepare_uploaded_image(file)

    # Prompt designed to extract visual label-value pairs without requiring OCR-B MRZ text
    prompt = (
        "You are an expert document analyzer specializing in international passport bio-data pages.\n"
        "Analyze the passport page image using ONLY visual text fields (even if the bottom MRZ line is cropped, blurry, or missing).\n"
        "Extract the following details accurately:\n"
        "1. Passport Number (Document Number, often in top-right or body).\n"
        "2. Surname / Family Name.\n"
        "3. Given Names / First and Middle Names.\n"
        "4. Combine Surname and Given Names formatted as 'SURNAME, GIVEN NAMES'.\n"
        "5. Nationality (Country name or 3-letter code).\n"
        "6. Date of Birth, Date of Issue, and Date of Expiry (normalize all dates to YYYY-MM-DD).\n"
        "7. Sex / Gender ('M', 'F', or 'X').\n"
        "8. Place of Birth (City / Country).\n"
        "Return ONLY a valid JSON object matching this schema:\n"
        "{\n"
        '  "is_passport_detected": true,\n'
        '  "passport_number": "string or null",\n'
        '  "surname": "string or null",\n'
        '  "given_names": "string or null",\n'
        '  "full_name": "string or null",\n'
        '  "nationality": "string or null",\n'
        '  "date_of_birth": "YYYY-MM-DD or null",\n'
        '  "sex": "string or null",\n'
        '  "place_of_birth": "string or null",\n'
        '  "date_of_issue": "YYYY-MM-DD or null",\n'
        '  "date_of_expiry": "YYYY-MM-DD or null",\n'
        '  "confidence_note": "string or null"\n'
        "}\n"
        "Do not include any conversational text or markdown codeblocks outside the JSON."
    )

    data = execute_vlm_query(image, prompt)

    # Construct and return validated Pydantic model
    return PassportBioExtractionResponse(
        is_passport_detected=data.get("is_passport_detected", False),
        passport_number=data.get("passport_number"),
        surname=data.get("surname"),
        given_names=data.get("given_names"),
        full_name=data.get("full_name") or f"{data.get('surname', '')} {data.get('given_names', '')}".strip() or None,
        nationality=data.get("nationality"),
        date_of_birth=data.get("date_of_birth"),
        sex=data.get("sex"),
        place_of_birth=data.get("place_of_birth"),
        date_of_issue=data.get("date_of_issue"),
        date_of_expiry=data.get("date_of_expiry"),
        confidence_note=data.get("confidence_note")
    )


@app.get("/health", tags=["System"])
def health():
    """Health check endpoint indicating active endpoints and runtime parameters."""
    return {
        "status": "online" if "model" in ml_state else "initializing",
        "device": settings.device,
        "model_id": settings.model_id,
        "configured_host": settings.app_host,
        "configured_port": settings.app_port,
        "endpoints": {
            "stamp_extraction": full_stamp_path,
            "visa_extraction": full_visa_path,
            "passport_bio_extraction": full_bio_path
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False
    )