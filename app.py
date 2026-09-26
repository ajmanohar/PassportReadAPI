import io
import json
import re
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from pydantic import BaseModel, Field
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ==============================================================================
# 1. Pydantic Models for Output Validation
# ==============================================================================

class VisaExtractionResponse(BaseModel):
    """
    Structured data model for visa page extraction.
    Matches standard Form-C fields (Field 4: Place of issue of visa, Visa number).
    """
    is_visa_detected: bool = Field(
        default=False, 
        description="True if a visa sticker or consular stamp is present on the page"
    )
    visa_number: Optional[str] = Field(
        default=None, 
        description="Extracted visa number (printed or handwritten)"
    )
    place_of_issue: Optional[str] = Field(
        default=None, 
        description="City or mission where the visa was issued"
    )
    date_of_issue: Optional[str] = Field(
        default=None, 
        description="Date of visa issue in YYYY-MM-DD format"
    )
    date_of_expiry: Optional[str] = Field(
        default=None, 
        description="Date of visa expiry in YYYY-MM-DD format"
    )
    visa_type: Optional[str] = Field(
        default=None, 
        description="Type of visa (e.g., Tourist, Business, Entry, e-Visa)"
    )
    handwritten_notes: Optional[str] = Field(
        default=None, 
        description="Any manual officer remarks, endorsements, or pen notes"
    )


class StampExtractionResponse(BaseModel):
    """
    Structured data model for arrival / landing stamp extraction.
    Matches Form-C Field 5 (Date of Arrival in India) and Field 12 (Port/City arrived).
    """
    is_stamp_detected: bool = Field(
        default=False, 
        description="True if an immigration ink stamp is detected"
    )
    arrival_date: Optional[str] = Field(
        default=None, 
        description="Arrival or landing date normalized to YYYY-MM-DD"
    )
    port_of_entry: Optional[str] = Field(
        default=None, 
        description="Immigration airport or seaport name (e.g., COCHIN AIRPORT, DEL, BOM)"
    )
    stamp_type: str = Field(
        default="ENTRY", 
        description="Detected stamp category: ENTRY / ARRIVAL or DEPARTURE / EXIT"
    )
    stay_permitted_until: Optional[str] = Field(
        default=None, 
        description="Handwritten or stamped stay permission limit if present"
    )


# ==============================================================================
# 2. Global State & Model Lifespan Management
# ==============================================================================

# Global container for model and processor to load once into memory at startup
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the Qwen2-VL-2B-Instruct model and processor into CPU RAM once on startup.
    Frees memory when the application shuts down.
    """
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    print(f"Loading vision model '{model_id}' into CPU memory... This may take a minute on first run.")

    # AutoProcessor handles dynamic visual tokenization and text encoding
    # We restrict min/max image pixels to keep CPU inference fast and prevent RAM exhaustion
    processor = AutoProcessor.from_pretrained(
        model_id, 
        min_pixels=256 * 28 * 28, 
        max_pixels=1024 * 28 * 28
    )

    # Load model weights using standard 32-bit float for CPU execution
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        device_map="cpu",
        low_cpu_mem_usage=True
    )

    ml_models["processor"] = processor
    ml_models["model"] = model
    print("Vision model and processor successfully loaded into CPU RAM.")

    yield

    # Clean up on shutdown
    ml_models.clear()
    print("Unloaded vision model from memory.")


# Initialize the FastAPI app with lifespan handler
app = FastAPI(
    title="Passport Stamp & Visa Capture API",
    description="Local open-source CPU service using Qwen2-VL for passport stamps and visas.",
    version="1.0.0",
    lifespan=lifespan
)


# ==============================================================================
# 3. Helper Functions: Image Preprocessing & Inference
# ==============================================================================

def prepare_image(upload_file: UploadFile) -> Image.Image:
    """
    Validates uploaded file, reads raw bytes, and converts to an RGB PIL Image.
    Resizes image proportionally if dimensions exceed 1280px to optimize CPU speed.
    """
    try:
        content = upload_file.file.read()
        image = Image.open(io.BytesIO(content))

        # Standardize color mode to RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Downscale oversized mobile captures to optimize standard CPU runtime
        max_dim = 1280
        if max(image.size) > max_dim:
            image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)

        return image
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(exc)}"
        )


def execute_vlm_inference(image: Image.Image, system_prompt: str) -> dict:
    """
    Runs the Qwen2-VL model on CPU with the specified image and instruction prompt.
    Parses and extracts the JSON block generated by the model.
    """
    processor = ml_models.get("processor")
    model = ml_models.get("model")

    if not model or not processor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision model is not loaded."
        )

    # Construct chat-style message payload required by Qwen2-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": system_prompt}
            ]
        }
    ]

    # Format prompt text using Jinja chat template
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Process visual tokens and inputs
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    # Generate output tokens on CPU (greedy decoding for deterministic answers)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False
        )

    # Slice output tokens to trim the input prompt tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw_output = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]

    # Parse JSON from model response (handles raw JSON or ```json ... ``` code blocks)
    try:
        json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return json.loads(raw_output)
    except Exception:
        # Fallback if the model produced unstructured text
        return {"raw_response": raw_output}


# ==============================================================================
# 4. Service Endpoints
# ==============================================================================

@app.post(
    "/extract/visa",
    response_model=VisaExtractionResponse,
    summary="Extract Visa Details & Numbers",
    tags=["Passport Services"]
)
async def extract_visa(
    file: UploadFile = File(..., description="Image of passport page containing a visa sticker or consular stamp")
):
    """
    Endpoint 1: Analyzes image to detect visa stickers or ink-stamped visas.
    Extracts visa number, place of issue, dates, and handwritten officer annotations[cite: 1, 2].
    """
    image = prepare_image(file)

    prompt = (
        "You are an immigration document OCR expert. Analyze this image of a passport visa page.\n"
        "Look for a visa sticker, e-Visa page, or consular ink-stamped visa.\n"
        "Extract the following fields and return ONLY a valid JSON object:\n"
        "{\n"
        '  "is_visa_detected": true or false,\n'
        '  "visa_number": "string or null",\n'
        '  "place_of_issue": "string or null",\n'
        '  "date_of_issue": "YYYY-MM-DD or null",\n'
        '  "date_of_expiry": "YYYY-MM-DD or null",\n'
        '  "visa_type": "string or null",\n'
        '  "handwritten_notes": "any officer handwriting or manual remarks, or null"\n'
        "}\n"
        "Do not include any text before or after the JSON."
    )

    result_json = execute_vlm_inference(image, prompt)

    # Standardize result through Pydantic
    return VisaExtractionResponse(
        is_visa_detected=result_json.get("is_visa_detected", False),
        visa_number=result_json.get("visa_number"),
        place_of_issue=result_json.get("place_of_issue"),
        date_of_issue=result_json.get("date_of_issue"),
        date_of_expiry=result_json.get("date_of_expiry"),
        visa_type=result_json.get("visa_type"),
        handwritten_notes=result_json.get("handwritten_notes")
    )


@app.post(
    "/extract/stamp",
    response_model=StampExtractionResponse,
    summary="Extract Landing / Entry Date & Port",
    tags=["Passport Services"]
)
async def extract_stamp(
    file: UploadFile = File(..., description="Image of passport page containing immigration ink stamps")
):
    """
    Endpoint 2: Analyzes image to locate rubber ink immigration entry stamps[cite: 1, 2].
    Extracts the arrival date and port of entry, ignoring departure/exit stamps[cite: 1, 2].
    """
    image = prepare_image(file)

    prompt = (
        "You are an immigration stamp inspection assistant.\n"
        "Examine the image for immigration ink stamps (rectangular or oval rubber stamps).\n"
        "Identify the ARRIVAL / ENTRY stamp (marked 'ARRIVED', 'ENTRY', or incoming arrow).\n"
        "Ignore EXIT / DEPARTURE stamps.\n"
        "Extract the details and return ONLY a valid JSON object:\n"
        "{\n"
        '  "is_stamp_detected": true or false,\n'
        '  "arrival_date": "arrival date normalized to YYYY-MM-DD or null",\n'
        '  "port_of_entry": "immigration checkpoint airport/city name like COCHIN AIRPORT, DEL, BOM or null",\n'
        '  "stamp_type": "ENTRY",\n'
        '  "stay_permitted_until": "date or period if handwritten or stamped, or null"\n'
        "}\n"
        "Do not include any text before or after the JSON."
    )

    result_json = execute_vlm_inference(image, prompt)

    return StampExtractionResponse(
        is_stamp_detected=result_json.get("is_stamp_detected", False),
        arrival_date=result_json.get("arrival_date"),
        port_of_entry=result_json.get("port_of_entry"),
        stamp_type=result_json.get("stamp_type", "ENTRY"),
        stay_permitted_until=result_json.get("stay_permitted_until")
    )


@app.get("/health", tags=["Health"])
def health_check():
    """Confirms the API service is alive and model is initialized."""
    model_ready = "model" in ml_models
    return {
        "status": "online" if model_ready else "initializing",
        "device": "cpu",
        "engine": "Qwen2-VL-2B-Instruct"
    }