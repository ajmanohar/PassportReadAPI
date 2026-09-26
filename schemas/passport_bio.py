"""
schemas/passport_bio.py
-----------------------
Pydantic data models for passport bio-data page extraction when 
the Machine Readable Zone (MRZ) is cut off, smudged, or missing entirely.

Extracts data from the Visual Inspection Zone (VIZ) to feed Form-C:
- Name of foreigner (Surname First) -> Form-C Field 1
- Nationality                      -> Form-C Field 3
- Passport Number & Expiry Dates   -> Guest Identification
"""

from typing import Optional
from pydantic import BaseModel, Field


class PassportBioExtractionResponse(BaseModel):
    """
    Structured response schema representing non-MRZ passport bio-data.
    """
    is_passport_detected: bool = Field(
        default=False,
        description="True if a valid passport identity/bio page was identified"
    )
    passport_number: Optional[str] = Field(
        default=None,
        description="Alphanumeric document/passport number extracted from header or page body"
    )
    surname: Optional[str] = Field(
        default=None,
        description="Holder's family name / surname (Forms Field 1 Surname)"
    )
    given_names: Optional[str] = Field(
        default=None,
        description="Holder's first and middle names"
    )
    full_name: Optional[str] = Field(
        default=None,
        description="Full concatenated name (Surname First) as requested by Form-C Field 1"
    )
    nationality: Optional[str] = Field(
        default=None,
        description="Country code or full nationality name (Form-C Field 3)"
    )
    date_of_birth: Optional[str] = Field(
        default=None,
        description="Date of birth normalized to YYYY-MM-DD"
    )
    sex: Optional[str] = Field(
        default=None,
        description="Gender / Sex marker: M, F, or X"
    )
    place_of_birth: Optional[str] = Field(
        default=None,
        description="City and/or country where the document holder was born"
    )
    date_of_issue: Optional[str] = Field(
        default=None,
        description="Passport issuance date normalized to YYYY-MM-DD"
    )
    date_of_expiry: Optional[str] = Field(
        default=None,
        description="Passport expiration date normalized to YYYY-MM-DD"
    )
    confidence_note: Optional[str] = Field(
        default=None,
        description="Inspection notes regarding glare, missing MRZ lines, or obscured fields"
    )