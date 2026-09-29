"""
Pydantic data schemas for ICAO Doc 9303 MRZ extraction via PassportEye & Tesseract OCR.
Defines response structures for health diagnostics and parsed passport records.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class HealthStatusResponse(BaseModel):
    """
    Health diagnostic model verifying Tesseract OCR binary availability on the host system.
    """
    # Overall service status
    status: str = Field(
        default="healthy",
        description="Health indicator of the MRZ sub-service"
    )

    # Boolean flag indicating whether the Tesseract executable was resolved
    tesseract_detected: bool = Field(
        ...,
        description="True if the Tesseract binary exists at the configured path"
    )

    # Path to the detected Tesseract binary or error notice
    tesseract_path: str = Field(
        ...,
        description="Resolved path of the tesseract binary on the host operating system"
    )


class PassportDataResponse(BaseModel):
    """
    Structured extraction payload representing parsed ICAO Doc 9303 passport bio-data.
    Tailored for Form-C immigration reporting and guest identity validation.
    """
    # Cryptographic checksum validation score
    valid_mrz: bool = Field(
        ...,
        description="True if all ICAO Doc 9303 check digits match checksum calculations"
    )

    # Type of document (e.g., 'P' for standard passport)
    document_type: Optional[str] = Field(
        default=None,
        description="Document type code according to ICAO standard"
    )

    # Issuing state or organization 3-letter code
    country: Optional[str] = Field(
        default=None,
        description="Three-letter ICAO country code of the issuing authority"
    )

    # Primary identifier (family name / surname)
    surname: Optional[str] = Field(
        default=None,
        description="Holder's primary identifier / family name"
    )

    # Secondary identifier (given names)
    names: Optional[str] = Field(
        default=None,
        description="Holder's secondary identifiers / given names"
    )

    # Unique passport / document number
    passport_number: Optional[str] = Field(
        default=None,
        description="Unique passport document identifier number"
    )

    # Nationality 3-letter code
    nationality: Optional[str] = Field(
        default=None,
        description="Three-letter ICAO country code of holder's nationality"
    )

    # Date of birth (YYMMDD format per MRZ standard)
    date_of_birth: Optional[str] = Field(
        default=None,
        description="Holder's date of birth in MRZ standard format (YYMMDD)"
    )

    # Sex / Gender ('M', 'F', or '<' / 'X')
    sex: Optional[str] = Field(
        default=None,
        description="Holder's sex ('M', 'F', or 'X')"
    )

    # Expiry date (YYMMDD format per MRZ standard)
    expiration_date: Optional[str] = Field(
        default=None,
        description="Document expiration date in MRZ standard format (YYMMDD)"
    )

    # Unparsed raw text extracted from the MRZ lines
    raw_mrz_text: Optional[str] = Field(
        default=None,
        description="Raw decoded string output directly from the optical recognition reader"
    )

    # Warning notifications, low confidence warnings, or validation issues
    warnings: List[str] = Field(
        default_factory=list,
        description="Diagnostic warning messages regarding MRZ legibility or checksum discrepancies"
    )