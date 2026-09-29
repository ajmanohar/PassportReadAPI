"""
Pydantic data schemas for synchronous Visa and OCI extraction using Qwen2-VL.
Standardizes extracted fields to use FieldResult with confidence metrics and source tracking.
"""

from typing import Optional
from pydantic import BaseModel, Field

# Reusable structured field result model
from schemas.job import FieldResult


class VisaResponse(BaseModel):
    """
    Standardized response structure for Visa and OCI document extraction.
    Returns per-field confidence scores and source provenance.
    """
    # Visa or OCI document number
    visa_number: FieldResult = Field(
        default_factory=FieldResult,
        description="Indian Visa number or OCI registration number"
    )

    # Visa category or class (Tourist, Business, Entry, OCI, etc.)
    visa_type: FieldResult = Field(
        default_factory=FieldResult,
        description="Type or class of the visa"
    )

    # Date visa was issued (YYYY-MM-DD)
    visa_issue_date: FieldResult = Field(
        default_factory=FieldResult,
        description="Date of visa issuance formatted as YYYY-MM-DD"
    )

    # Date visa expires (YYYY-MM-DD)
    visa_expiry_date: FieldResult = Field(
        default_factory=FieldResult,
        description="Date of visa expiration formatted as YYYY-MM-DD"
    )

    # Place or mission where visa was granted
    visa_place_of_issue: FieldResult = Field(
        default_factory=FieldResult,
        description="City or embassy where the visa was issued"
    )

    # Explanatory note from vision model regarding document readability
    confidence_note: Optional[str] = Field(
        default=None,
        description="Qualitative assessment of visual legibility and clarity"
    )

    # Execution status indicator string
    status: str = Field(
        default="success",
        description="Execution status of the visa extraction pipeline"
    )