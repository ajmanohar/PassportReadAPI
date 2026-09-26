"""
schemas/visa.py
---------------
Pydantic response models for passport visa page extraction.
Directly aligns with Form-C fields:
- place_of_issue -> Form-C Field 4: 'Place of issue of visa'
- visa_number    -> Form-C visa identification number
"""

from typing import Optional
from pydantic import BaseModel, Field

class VisaExtractionResponse(BaseModel):
    """
    Structured extraction response for visa pages (stickers or rubber consular stamps).
    """
    is_visa_detected: bool = Field(
        default=False,
        description="True if a valid visa sticker, e-Visa, or consular ink stamp was found"
    )
    visa_number: Optional[str] = Field(
        default=None,
        description="Extracted alphanumeric visa number (printed or handwritten)"
    )
    place_of_issue: Optional[str] = Field(
        default=None,
        description="City, mission, or consulate where the visa was issued (Form-C Field 4)"
    )
    date_of_issue: Optional[str] = Field(
        default=None,
        description="Visa issue date normalized to YYYY-MM-DD"
    )
    date_of_expiry: Optional[str] = Field(
        default=None,
        description="Visa expiration date normalized to YYYY-MM-DD"
    )
    visa_type: Optional[str] = Field(
        default=None,
        description="Category of visa (e.g., Tourist, Business, Entry, e-Visa, OCI)"
    )
    entries_allowed: Optional[str] = Field(
        default=None,
        description="Number of entries allowed: Single (S), Double (D), or Multiple (M)"
    )
    handwritten_notes: Optional[str] = Field(
        default=None,
        description="Any manual officer remarks, endorsements, or pen-written notes"
    )
    confidence_note: Optional[str] = Field(
        default=None,
        description="Notes regarding ink darkness, blur, overlapping marks, or legibility"
    )