"""
schemas/stamp.py
----------------
Pydantic data models specifically designed for immigration entry stamps.
Directly aligns with Form-C fields:
- arrival_date -> Form-C Field 5: 'Date of Arrival in India'
- port_of_entry -> Form-C Field 12: 'Country & city you arrived to India'
"""

from typing import Optional
from pydantic import BaseModel, Field

class StampExtractionResponse(BaseModel):
    """
    Structured extraction schema for immigration entry/landing stamps.
    """
    is_stamp_detected: bool = Field(
        default=False,
        description="True if an immigration ink stamp was found on the page image"
    )
    arrival_date: Optional[str] = Field(
        default=None,
        description="Landing or entry date normalized to YYYY-MM-DD"
    )
    port_of_entry: Optional[str] = Field(
        default=None,
        description="Immigration airport or port checkpoint (e.g., COCHIN AIRPORT, DEL, BOM)"
    )
    stamp_type: str = Field(
        default="ENTRY",
        description="Detected stamp classification: ENTRY / ARRIVAL or DEPARTURE / EXIT"
    )
    stay_permitted_until: Optional[str] = Field(
        default=None,
        description="Stay duration or expiry date if handwritten or stamped by the officer"
    )
    confidence_note: Optional[str] = Field(
        default=None,
        description="Notes regarding ink darkness, stamp angle, blur, or legibility"
    )