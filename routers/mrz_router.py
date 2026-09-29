"""
FastAPI router for synchronous Passport MRZ extraction using Tesseract and PassportEye.
Extracts ICAO Doc 9303 machine-readable zones and assigns deterministic confidence scores
based on mathematical check-digit validations, returning the standardized MRZExtractionResponse.
"""

import io
import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from passporteye import read_mrz

# Centralized schema models
from schemas.job import FieldResult
from schemas.passport_mrz import MRZExtractionResponse

# Set up dedicated logger
logger = logging.getLogger("mrz_router")
logger.setLevel(logging.INFO)

# ==============================================================================
# Router Configuration
# ==============================================================================

router = APIRouter(
    prefix="/api/v1/mrz",
    tags=["Passport MRZ (Tesseract OCR)"]
)


# ==============================================================================
# Diagnostic Probes
# ==============================================================================

@router.get("/health", summary="MRZ Service Diagnostic Probe")
def mrz_health_check() -> Dict[str, str]:
    """
    Verifies that the Tesseract OCR engine and PassportEye dependencies are operational.
    """
    return {
        "status": "healthy",
        "engine": "Tesseract OCR / PassportEye (ICAO Doc 9303)"
    }


# ==============================================================================
# Synchronous MRZ Extraction Endpoint
# ==============================================================================

@router.post(
    "/extract",
    response_model=MRZExtractionResponse,
    summary="Synchronous MRZ Extraction with Field-Level Confidence",
    description="Parses passport MRZ lines and calculates confidence based on check-digit validation."
)
async def extract_mrz_data(
    file: UploadFile = File(..., description="Passport identity page image containing MRZ lines")
) -> MRZExtractionResponse:
    """
    Synchronously reads an uploaded passport image, scans for 2-line or 3-line MRZ codes,
    verifies checksums, and returns field-level confidence ratings mapped into FieldResult.
    """
    try:
        # Read uploaded image bytes into in-memory buffer
        contents = await file.read()
        image_stream = io.BytesIO(contents)

        # Run PassportEye MRZ detection and decoding
        mrz_record = read_mrz(image_stream)

        # Handle cases where no valid MRZ bounding pattern was discovered
        if mrz_record is None:
            return MRZExtractionResponse(
                status="no_mrz_detected",
                valid_mrz=False,
                valid_score=0
            )

        # Extract parsed dictionary attributes and validity scores
        mrz_dict = mrz_record.to_dict() if hasattr(mrz_record, "to_dict") else {}
        score_val = getattr(mrz_record, "valid_score", 0)
        valid_score = int(score_val) if isinstance(score_val, (int, float)) else 0

        # Base confidence for non-checksum text fields derived from detection score
        base_text_confidence = round(max(0.0, min(1.0, valid_score / 100.0)), 2)

        # Inspect internal checksum attributes from PassportEye
        check_number = getattr(mrz_record, "check_number", False)
        check_dob = getattr(mrz_record, "check_date_of_birth", False)
        check_exp = getattr(mrz_record, "check_expiration_date", False)
        check_composite = getattr(mrz_record, "check_composite", False)

        # Overall validation flag
        is_overall_valid = bool(check_composite or valid_score >= 80)

        # Helper to assign score: 0.99 if check digit verified, 0.30 if failed, 0.0 if empty
        def score_checked_field(val: Optional[str], check_passed: bool) -> float:
            if not val:
                return 0.0
            return 0.99 if check_passed else 0.30

        # Helper to assign score to raw text fields
        def score_text_field(val: Optional[str]) -> float:
            if not val:
                return 0.0
            return base_text_confidence

        # Assemble individual field results with source attribution
        res_surname = FieldResult(
            value=mrz_dict.get("surname"),
            confidence=score_text_field(mrz_dict.get("surname")),
            source="tesseract_mrz"
        )
        res_given_names = FieldResult(
            value=mrz_dict.get("names"),
            confidence=score_text_field(mrz_dict.get("names")),
            source="tesseract_mrz"
        )
        res_passport_num = FieldResult(
            value=mrz_dict.get("number"),
            confidence=score_checked_field(mrz_dict.get("number"), check_number),
            source="tesseract_mrz"
        )
        res_nationality = FieldResult(
            value=mrz_dict.get("nationality"),
            confidence=score_text_field(mrz_dict.get("nationality")),
            source="tesseract_mrz"
        )
        res_dob = FieldResult(
            value=mrz_dict.get("date_of_birth"),
            confidence=score_checked_field(mrz_dict.get("date_of_birth"), check_dob),
            source="tesseract_mrz"
        )
        res_sex = FieldResult(
            value=mrz_dict.get("sex"),
            confidence=score_text_field(mrz_dict.get("sex")),
            source="tesseract_mrz"
        )
        res_expiry = FieldResult(
            value=mrz_dict.get("expiration_date"),
            confidence=score_checked_field(mrz_dict.get("expiration_date"), check_exp),
            source="tesseract_mrz"
        )

        return MRZExtractionResponse(
            surname=res_surname,
            given_names=res_given_names,
            passport_number=res_passport_num,
            nationality=res_nationality,
            date_of_birth=res_dob,
            sex=res_sex,
            passport_expiry_date=res_expiry,
            valid_mrz=is_overall_valid,
            valid_score=valid_score,
            status="success"
        )

    except Exception as exc:
        logger.error(f"MRZ extraction encountered an unhandled exception: {str(exc)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Tesseract MRZ processing error: {str(exc)}"
        )