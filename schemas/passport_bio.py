"""
Pydantic data schemas for synchronous Passport Bio-Data visual extraction via Qwen2-VL.
Standardizes every extracted field to use FieldResult with confidence metrics and source attribution.
"""

from typing import Optional
from pydantic import BaseModel, Field

# Reusable structured field result model
from schemas.job import FieldResult


class PassportBioResponse(BaseModel):
    """
    Standardized synchronous response structure for Passport Bio page visual extraction.
    Returns per-field confidence scores (0.0 to 1.0) and source provenance ('qwen2_vl').
    """
    # Holder's surname / family name
    surname: FieldResult = Field(
        default_factory=FieldResult,
        description="Holder's surname or family name"
    )

    # Holder's secondary given names
    given_names: FieldResult = Field(
        default_factory=FieldResult,
        description="Holder's given names"
    )

    # Unique passport identifier number
    passport_number: FieldResult = Field(
        default_factory=FieldResult,
        description="Passport document identification number"
    )

    # Holder's nationality (3-letter country code or full name)
    nationality: FieldResult = Field(
        default_factory=FieldResult,
        description="Country of citizenship or nationality"
    )

    # Date of birth (YYYY-MM-DD or document string)
    date_of_birth: FieldResult = Field(
        default_factory=FieldResult,
        description="Date of birth formatted as YYYY-MM-DD"
    )

    # Biological sex code ('M', 'F', or 'X')
    sex: FieldResult = Field(
        default_factory=FieldResult,
        description="Sex code ('M', 'F', or 'X')"
    )

    # Document expiration date (YYYY-MM-DD)
    passport_expiry_date: FieldResult = Field(
        default_factory=FieldResult,
        description="Passport expiration date formatted as YYYY-MM-DD"
    )

    # Model explanation regarding visual legibility or image artifacts
    confidence_note: Optional[str] = Field(
        default=None,
        description="Qualitative assessment of image clarity and document legibility"
    )

    # Execution status indicator
    status: str = Field(
        default="success",
        description="Execution status of the visual extraction pipeline"
    )