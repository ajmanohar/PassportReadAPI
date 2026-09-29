"""
Pydantic data schemas for synchronous Passport MRZ extraction using Tesseract and PassportEye.
Standardizes every extracted field to use FieldResult with check-digit backed confidence scores
and source attribution.
"""

from typing import Optional
from pydantic import BaseModel, Field

# Reusable structured field result model
from schemas.job import FieldResult


class MRZExtractionResponse(BaseModel):
    """
    Standardized synchronous response structure for Passport MRZ extraction.
    Returns per-field confidence scores derived from ICAO Doc 9303 modulo-10 checksum validation
    and source provenance ('tesseract_mrz').
    """
    # Holder's family surname
    surname: FieldResult = Field(
        default_factory=FieldResult,
        description="Holder's surname or family name extracted from MRZ line 1"
    )

    # Holder's secondary given names
    given_names: FieldResult = Field(
        default_factory=FieldResult,
        description="Holder's given names extracted from MRZ line 1"
    )

    # Unique passport identification code (checksum protected)
    passport_number: FieldResult = Field(
        default_factory=FieldResult,
        description="Passport document identification number verified against check digit"
    )

    # 3-letter ICAO country code of nationality
    nationality: FieldResult = Field(
        default_factory=FieldResult,
        description="3-letter ICAO country code of holder's nationality"
    )

    # Date of birth (YYMMDD format from MRZ, checksum protected)
    date_of_birth: FieldResult = Field(
        default_factory=FieldResult,
        description="Holder's date of birth in YYMMDD format verified against check digit"
    )

    # Biological sex code ('M', 'F', or '<')
    sex: FieldResult = Field(
        default_factory=FieldResult,
        description="Biological sex indicator code ('M', 'F', or 'X')"
    )

    # Document expiration date (YYMMDD format from MRZ, checksum protected)
    passport_expiry_date: FieldResult = Field(
        default_factory=FieldResult,
        description="Passport expiration date in YYMMDD format verified against check digit"
    )

    # True if overall composite checksum matches ICAO Doc 9303 specifications
    valid_mrz: bool = Field(
        default=False,
        description="True if all mathematical check digits and composite checks match"
    )

    # Raw recognition confidence score (0 to 100) provided by PassportEye
    valid_score: int = Field(
        default=0,
        description="PassportEye internal OCR detection and character match quality score (0 - 100)"
    )

    # Execution status indicator string
    status: str = Field(
        default="success",
        description="Execution status of the MRZ detection pipeline ('success' or 'no_mrz_detected')"
    )