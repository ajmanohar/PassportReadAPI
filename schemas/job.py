"""
Pydantic data schemas for asynchronous batch document processing.
Defines data structures for job creation, polling status, and fused Form-C extraction payloads.
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
    Directly aligns with Bureau of Immigration Form-C fields for guest registration.
    """
    # --- Passport & Bio-Data (MRZ primary, VLM fallback) ---
    surname: Optional[str] = Field(default=None, description="Holder's surname / family name")
    given_names: Optional[str] = Field(default=None, description="Holder's given names / secondary identifiers")
    sex: Optional[str] = Field(default=None, description="Sex code ('M', 'F', or 'X')")
    date_of_birth: Optional[str] = Field(default=None, description="Date of birth in YYYY-MM-DD or YYMMDD format")
    nationality: Optional[str] = Field(default=None, description="Holder's 3-letter ICAO country code of nationality")
    passport_number: Optional[str] = Field(default=None, description="Unique passport identifier number")
    passport_expiry_date: Optional[str] = Field(default=None, description="Passport expiration date")
    valid_mrz: bool = Field(default=False, description="True if ICAO Doc 9303 checksum validation succeeded")

    # --- Visa / OCI Details ---
    visa_number: Optional[str] = Field(default=None, description="Indian Visa number or OCI registration number")
    visa_type: Optional[str] = Field(default=None, description="Visa category/type (Tourist, Business, OCI, etc.)")
    visa_issue_date: Optional[str] = Field(default=None, description="Visa issue date (YYYY-MM-DD)")
    visa_expiry_date: Optional[str] = Field(default=None, description="Visa expiry date (YYYY-MM-DD)")
    visa_place_of_issue: Optional[str] = Field(default=None, description="City / authority where visa was issued")

    # --- Immigration Entry Stamp Details ---
    arrival_date_india: Optional[str] = Field(default=None, description="Date entered India from immigration stamp (YYYY-MM-DD)")
    arrival_port_india: Optional[str] = Field(default=None, description="Port or airport of entry into India")

    # --- Handwritten Welcome Form Details ---
    contact_phone: Optional[str] = Field(default=None, description="Guest phone/mobile number")
    contact_email: Optional[str] = Field(default=None, description="Guest email address")
    permanent_address: Optional[str] = Field(default=None, description="Permanent home address in home country")
    arrived_from: Optional[str] = Field(default=None, description="Place or city arrived from before check-in")
    proceeding_to: Optional[str] = Field(default=None, description="Destination proceeding to next")
    purpose_of_visit: Optional[str] = Field(default=None, description="Purpose of visit and profession")

    # --- Pipeline Diagnostics ---
    processing_notes: List[str] = Field(
        default_factory=list,
        description="Diagnostic notes or confidence flags logged across extraction stages"
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
        description="Unified Form-C guest record once processing succeeds"
    )

    # Error explanation (populated only if status is FAILED)
    error_message: Optional[str] = Field(
        default=None,
        description="Details of unhandled execution failure if processing aborted"
    )