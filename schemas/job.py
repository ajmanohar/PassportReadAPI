"""
Pydantic data schemas for asynchronous batch document processing.
Defines data structures for job creation, polling status, and fused Form-C extraction payloads
with per-field confidence scores and extraction provenance tracking.
Fully aligned with Bureau of Immigration Form-C (Rule 14) Arrival Report fields.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """
    Lifecycle states for an asynchronous document processing job.
    """
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class FieldResult(BaseModel):
    """
    Represents an extracted field value paired with a normalized confidence score
    and source attribution (e.g., 'tesseract_mrz', 'qwen2_vl', or 'welcome_form').
    """
    # The extracted text value (string or None)
    value: Optional[str] = Field(
        default=None,
        description="Extracted text content for the field"
    )

    # Confidence score normalized between 0.0 (unreliable) and 1.0 (verified)
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for this specific field between 0.0 and 1.0"
    )

    # Origin of the winning value ('tesseract_mrz', 'qwen2_vl', 'welcome_form', or 'unassigned')
    source: str = Field(
        default="unassigned",
        description="The engine or pipeline stage that produced this winning value"
    )


class JobSubmissionResponse(BaseModel):
    """
    Immediate acknowledgement response returned when document images are ingested.
    Provides tracking UUID for asynchronous status polling.
    """
    # Unique tracking identifier for polling job completion
    job_id: str = Field(
        ...,
        description="Unique UUID tracking identifier for the background processing task"
    )

    # Initial queued state
    status: JobStatus = Field(
        default=JobStatus.QUEUED,
        description="Current state of the asynchronous pipeline"
    )

    # ISO timestamp when the job was accepted
    created_at: str = Field(
        ...,
        description="ISO 8601 formatted timestamp of job intake"
    )

    # Diagnostic message
    message: str = Field(
        default="Documents queued for extraction. Poll /api/v1/jobs/{job_id} for completion.",
        description="Human-readable status notice"
    )


class GuestFormCRecord(BaseModel):
    """
    Unified, fused bio-data model merging MRZ, VLM bio, Visa, Stamp, and Welcome Form data.
    Every individual field carries its own value, confidence score (0.0 - 1.0), and source engine.
    Matches all 12 items of the Bureau of Immigration Form-C Arrival Report (Rule 14).
    """
    # --- Item 1 & Passport Bio-Data ---
    surname: FieldResult = Field(default_factory=FieldResult, description="Holder's surname / family name")
    given_names: FieldResult = Field(default_factory=FieldResult, description="Holder's given names")
    sex: FieldResult = Field(default_factory=FieldResult, description="Sex code ('M', 'F', or 'X')")
    date_of_birth: FieldResult = Field(default_factory=FieldResult, description="Date of birth (YYYY-MM-DD)")
    nationality: FieldResult = Field(default_factory=FieldResult, description="Holder's country code or nationality")
    passport_number: FieldResult = Field(default_factory=FieldResult, description="Unique passport identifier number")
    passport_expiry_date: FieldResult = Field(default_factory=FieldResult, description="Passport expiration date (YYYY-MM-DD)")
    valid_mrz: bool = Field(default=False, description="True if ICAO Doc 9303 checksum validation succeeded")

    # --- Item 2: Guest Contact & Address Details ---
    permanent_address: FieldResult = Field(default_factory=FieldResult, description="Full residential address")
    city: FieldResult = Field(default_factory=FieldResult, description="Home city")
    country: FieldResult = Field(default_factory=FieldResult, description="Home country")
    contact_email: FieldResult = Field(default_factory=FieldResult, description="Guest email address")
    contact_phone: FieldResult = Field(default_factory=FieldResult, description="Guest telephone / mobile number")

    # --- Items 3 & 4: Visa / OCI Details ---
    visa_number: FieldResult = Field(default_factory=FieldResult, description="Indian Visa number or OCI registration number")
    visa_type: FieldResult = Field(default_factory=FieldResult, description="Visa category/type (Tourist, Business, OCI, etc.)")
    visa_issue_date: FieldResult = Field(default_factory=FieldResult, description="Visa issue date (YYYY-MM-DD)")
    visa_expiry_date: FieldResult = Field(default_factory=FieldResult, description="Visa expiry date (YYYY-MM-DD)")
    visa_place_of_issue: FieldResult = Field(default_factory=FieldResult, description="City / authority where visa was issued")

    # --- Item 5: Immigration Entry Stamp ---
    arrival_date_india: FieldResult = Field(default_factory=FieldResult, description="Date of arrival in India (YYYY-MM-DD)")
    arrival_port_india: FieldResult = Field(default_factory=FieldResult, description="Port or airport of entry into India")

    # --- Items 6 - 12: Homestay Arrival & Movement (Form-C Rule 14) ---
    arrival_datetime_homestay: FieldResult = Field(
        default_factory=FieldResult,
        description="Item 6: Date and time of arrival in the home stay"
    )
    arrived_from: FieldResult = Field(
        default_factory=FieldResult,
        description="Item 7: City / place arrived from"
    )
    proceeding_to: FieldResult = Field(
        default_factory=FieldResult,
        description="Item 8: Destination proceeding to next"
    )
    stay_duration_india: FieldResult = Field(
        default_factory=FieldResult,
        description="Item 9: Proposed duration of stay in India"
    )
    purpose_of_visit: FieldResult = Field(
        default_factory=FieldResult,
        description="Item 10: Purpose of visit and profession"
    )
    expected_departure_date: FieldResult = Field(
        default_factory=FieldResult,
        description="Item 11: Expected date of departure from the homestay"
    )
    arrived_country_city_india: FieldResult = Field(
        default_factory=FieldResult,
        description="Item 12: Country & city arrived to India"
    )

    # --- Pipeline Diagnostics ---
    processing_notes: List[str] = Field(
        default_factory=list,
        description="Diagnostic notes or confidence comparison logs across extraction stages"
    )


class JobDetailResponse(BaseModel):
    """
    Detailed status and result payload returned upon polling /api/v1/jobs/{job_id}.
    """
    # Unique tracking identifier
    job_id: str = Field(..., description="UUID tracking identifier")

    # Current lifecycle state (QUEUED, PROCESSING, COMPLETED, FAILED)
    status: JobStatus = Field(..., description="Current job status")

    # ISO timestamps tracking execution timeline
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    updated_at: Optional[str] = Field(default=None, description="ISO 8601 last modified timestamp")

    # Final fused extraction data (populated only when status is COMPLETED)
    result: Optional[GuestFormCRecord] = Field(
        default=None,
        description="Unified Form-C guest record with field-level confidence once processing succeeds"
    )

    # Error explanation (populated only if status is FAILED)
    error_message: Optional[str] = Field(
        default=None,
        description="Details of unhandled execution failure if processing aborted"
    )