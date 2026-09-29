"""
Modular API router for ICAO Doc 9303 Machine Readable Zone (MRZ) extraction.
Wraps PassportEye and Tesseract OCR into an asynchronous FastAPI endpoint,
validating checksums and generating structured bio-data models for Form-C immigration pipelines.
"""

import io
import os
from typing import Dict, Any, List

from fastapi import APIRouter, File, UploadFile, HTTPException, status
import pytesseract
from passporteye import read_mrz

# Central configuration and schema imports
from config import settings
from schemas.passport_mrz import HealthStatusResponse, PassportDataResponse


# ==============================================================================
# APIRouter Definition
# ==============================================================================

router = APIRouter(
    prefix=settings.mrz_router_prefix,
    tags=["Passport MRZ (Tesseract OCR)"]
)


# ==============================================================================
# Tesseract Runtime Environment Initialization
# ==============================================================================

def initialize_tesseract_environment() -> None:
    """
    Configures pytesseract command pointers, sets TESSDATA_PREFIX for language models,
    and prepends the Tesseract binary folder to system PATH so PassportEye subprocesses succeed.
    """
    # 1. Assign explicit binary command location to pytesseract
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

    # 2. Assign tessdata environment variable if directory exists
    if os.path.exists(settings.tessdata_prefix):
        os.environ["TESSDATA_PREFIX"] = settings.tessdata_prefix

    # 3. Prepend tesseract directory to host PATH so subprocess executions locate it
    tesseract_dir = os.path.dirname(settings.tesseract_cmd)
    current_path = os.environ.get("PATH", "")
    if os.path.exists(tesseract_dir) and tesseract_dir not in current_path:
        os.environ["PATH"] = tesseract_dir + os.pathsep + current_path


# Execute environment variable configuration on module import
initialize_tesseract_environment()


# ==============================================================================
# Diagnostic Probes
# ==============================================================================

@router.get(
    "/health",
    response_model=HealthStatusResponse,
    summary="Health check for Tesseract OCR engine",
    tags=["Passport MRZ (Tesseract OCR)"]
)
def tesseract_health_check() -> HealthStatusResponse:
    """
    Verifies that the configured Tesseract executable is present on the host filesystem
    without leaking internal host paths to public callers.
    """
    binary_found = os.path.isfile(settings.tesseract_cmd)
    return HealthStatusResponse(
        status="healthy" if binary_found else "degraded",
        ocr_engine_ready=binary_found
    )


# ==============================================================================
# MRZ Extraction Endpoint
# ==============================================================================

@router.post(
    "/extract",
    response_model=PassportDataResponse,
    summary="Extract and validate ICAO Doc 9303 MRZ from passport image",
    tags=["Passport MRZ (Tesseract OCR)"]
)
async def extract_passport_mrz(
    file: UploadFile = File(..., description="Image file of the passport identity bio page containing MRZ bands")
) -> PassportDataResponse:
    """
    Consumes an uploaded passport photo, extracts the two or three ICAO Doc 9303 MRZ lines,
    evaluates cryptographic checksum validity, and maps fields into structured passport data.
    """
    # Verify incoming payload MIME type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported media type: '{file.content_type}'. Please upload a valid image file."
        )

    # Read binary bytes from multipart stream
    try:
        image_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file stream: {str(exc)}"
        )

    # Execute PassportEye optical parsing over memory stream
    try:
        mrz_record = read_mrz(io.BytesIO(image_bytes))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optical character recognition execution failure: {str(exc)}"
        )

    # Handle cases where no MRZ was detected in the image
    if mrz_record is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unable to locate or decode MRZ. Ensure bottom lines are sharp, glare-free, and unobscured."
        )

    # Extract dictionary representation and validation score
    mrz_data: Dict[str, Any] = mrz_record.to_dict() if hasattr(mrz_record, "to_dict") else {}
    score = getattr(mrz_record, "valid_score", 0)

    # Score of 80+ indicates valid ICAO checksums
    is_valid = bool(score >= 80) if isinstance(score, (int, float)) else bool(score)

    # Compile diagnostic warnings if check digits fail
    warnings: List[str] = []
    if not is_valid:
        warnings.append("Checksum score is low or invalid. Manually verify output against source passport image.")

    return PassportDataResponse(
        valid_mrz=is_valid,
        document_type=mrz_data.get("type"),
        country=mrz_data.get("country"),
        surname=mrz_data.get("surname"),
        names=mrz_data.get("names"),
        passport_number=mrz_data.get("number"),
        nationality=mrz_data.get("nationality"),
        date_of_birth=mrz_data.get("date_of_birth"),
        sex=mrz_data.get("sex"),
        expiration_date=mrz_data.get("expiration_date"),
        raw_mrz_text=getattr(mrz_record, "raw_text", None),
        warnings=warnings
    )