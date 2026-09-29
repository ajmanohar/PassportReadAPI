"""
Pydantic data schemas for synchronous Immigration Entry Stamp extraction using Qwen2-VL.
Standardizes extracted fields to use FieldResult with confidence metrics and source tracking.
"""

from typing import Optional
from pydantic import BaseModel, Field

# Reusable structured field result model
from schemas.job import FieldResult


class StampResponse(BaseModel):
    """
    Standardized response structure for immigration arrival ink stamp extraction.
    Returns per-field confidence scores and source provenance.
    """
    # Date of arrival stamped by immigration officials (YYYY-MM-DD)
    arrival_date_india: FieldResult = Field(
        default_factory=FieldResult,
        description="Date of entry into India formatted as YYYY-MM-DD"
    )

    # Port or airport code/name of entry
    arrival_port_india: FieldResult = Field(
        default_factory=FieldResult,
        description="Port or checkpoint of entry into India"
    )

    # Explanatory note from vision model regarding stamp contrast or ink smudging
    confidence_note: Optional[str] = Field(
        default=None,
        description="Qualitative assessment of stamp clarity and ink legibility"
    )

    # Execution status indicator string
    status: str = Field(
        default="success",
        description="Execution status of the stamp extraction pipeline"
    )