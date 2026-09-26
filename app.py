"""
app.py
------
Main application service for passport immigration stamp capture.
Reads host, port, and route paths dynamically from config.py.
Executes Qwen2-VL locally on standard CPU.
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

# Import externalized configuration and schemas
from config import settings
from schemas.stamp import StampExtractionResponse

# ==============================================================================
# Model Lifespan & Global State
# ==============================================================================

# Dictionary to hold the loaded model and tokenizer in RAM
ml_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager:
    Loads the Hugging Face vision model into CPU RAM once at application launch.
    Ensures memory is cleaned up when shutting down.
    """
    print(f"[*] Initializing model '{settings.model_id}' on {settings.device.upper()}...")
    
    # AutoProcessor automatically adapts visual token patching
    processor = AutoProcessor.from_pretrained(
        settings.model_id,
        min_pixels=256 * 28 * 28,
        max_pixels=1024 * 28 * 28
    )

    # Load model weights targeting CPU execution
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        settings.model_id,
        torch_dtype=torch.float32,
        device_map=settings.device,
        low_cpu_mem_usage=True
    )

    # Store handles in global state
    ml_state["processor"] = processor
    ml_state["model"] = model
    print("[*] Vision-Language Model loaded and ready.")

    yield

    # Clean up upon server termination
    ml_state.clear()
    print("[*] Freed model resources from RAM.")


# ==============================================================================
# FastAPI Application Initialization
# ==============================================================================

app = FastAPI(
    title="Passport Stamp Capture API",
    description="Microservice for extracting arrival dates and ports of entry from passport ink stamps.",
    version="1.0.0",
    lifespan=lifespan
)


# ==============================================================================
# Image Processing Helper
# ==============================================================================

def prepare_uploaded_image(upload_file: UploadFile) -> Image.Image:
    """
    Reads the uploaded image bytes, converts to RGB, and scales down oversized
    dimensions according to config.py to ensure low inference latency on standard CPU.
    """
    try:
        raw_bytes = upload_file.file.read()
        image = Image.open(io.BytesIO(raw_bytes))

        # Ensure uniform RGB channels (converts RGBA, CMYK, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Downscale if the image exceeds the configured threshold
        max_dim = settings.max_image_dimension
        if max(image.size) > max_dim:
            image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        return image
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to process image: {str(exc)}"
        )


# ==============================================================================
# Inference Handler
# ==============================================================================

def run_stamp_inference(image: Image.Image) -> dict:
    """
    Executes Qwen2-VL vision inference on CPU and extracts the structured JSON response.
    """
    processor = ml_state.get("processor")
    model = ml_state.get("model")

    if not model or not processor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not initialized."
        )

    # Prompt engineered to differentiate entry stamps from exit stamps and extract Form-C fields
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

    # Prepare chat format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Preprocess prompt and visual patch tensors
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    # Perform inference on CPU
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False  # Deterministic output
        )

    # Decode tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Parse JSON regex match
    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(raw_text)
    except Exception:
        return {"is_stamp_detected": False, "confidence_note": f"Raw output: {raw_text}"}


# ==============================================================================
# API Endpoints (Path read dynamically from config)
# ==============================================================================

# Construct the full endpoint path dynamically from settings
full_stamp_path = f"{settings.api_v1_prefix}{settings.stamp_extract_path}"

@app.post(
    full_stamp_path,
    response_model=StampExtractionResponse,
    summary="Extract Arrival Date and Port from Immigration Ink Stamp",
    tags=["Stamp Extraction"]
)
async def extract_stamp_data(
    file: UploadFile = File(..., description="Passport page image containing ink immigration stamps")
):
    """
    Scans the uploaded passport page for immigration arrival stamps,
    extracting the landing date and port of entry for Form-C reporting[cite: 1, 2].
    """
    image = prepare_uploaded_image(file)
    extracted_data = run_stamp_inference(image)
    
    return StampExtractionResponse(
        is_stamp_detected=extracted_data.get("is_stamp_detected", False),
        arrival_date=extracted_data.get("arrival_date"),
        port_of_entry=extracted_data.get("port_of_entry"),
        stamp_type=extracted_data.get("stamp_type", "ENTRY"),
        stay_permitted_until=extracted_data.get("stay_permitted_until"),
        confidence_note=extracted_data.get("confidence_note")
    )


@app.get("/health", tags=["System"])
def health():
    """Health check endpoint displaying server runtime parameters."""
    return {
        "status": "online" if "model" in ml_state else "initializing",
        "configured_port": settings.app_port,
        "configured_host": settings.app_host,
        "stamp_endpoint": full_stamp_path,
        "device": settings.device
    }


# ==============================================================================
# Application Entry Point
# ==============================================================================

if __name__ == "__main__":
    # Runs uvicorn using configuration values defined in settings
    uvicorn.run(
        "app:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False
    )