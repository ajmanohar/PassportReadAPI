"""
==============================================================================
schemas/job.py - PassportReadAPI
==============================================================================
Pydantic data schemas for asynchronous batch document processing.
Defines data structures for job creation, polling status, and multi-document
extractions with per-field confidence scores and engine provenance tracking.

Architecture:
- Isolates raw extraction data per document source:
    * passport_bio   : MRZ + VLM bio page data
    * visa_document  : e-Visa paper / PDF document scan
    * visa_stamp     : Passport booklet consular visa stamp
    * entry_stamp    : Bureau of Immigration rubber ink entry stamp
    * welcome_form   : Form-C Rule 14 handwritten arrival report
- Maintains fused_record for final Form-C portal filing.
- Preserves full backward compatibility for existing mobile and desktop clients.
==============================================================================
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """
    Lifecycle execution states for an asynchronous document processing job.
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
    # The extracted text value (string or None if unextracted/missing)
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

    # Origin provenance ('tesseract_mrz', 'qwen2_vl', 'welcome_form', or 'unassigned')
    source: str = Field(
        default="unassigned",
        description="The engine or pipeline stage that produced this value"
    )


# ==============================================================================
# Isolated Per-Document Extraction Models (Task 45)
# ==============================================================================

class PassportBioRecord(BaseModel):
    """
    Isolated extraction attributes from Passport Bio-Data page (MRZ + VLM).
    """
    surname: FieldResult = Field(default_factory=FieldResult, description="Holder surname")
    given_names: FieldResult = Field(default_factory=FieldResult, description="Holder given names")
    sex: FieldResult = Field(default_factory=FieldResult, description="Sex code ('M', 'F', 'X')")
    date_of_birth: FieldResult = Field(default_factory=FieldResult, description="Date of birth (YYYY-MM-DD)")
    nationality: FieldResult = Field(default_factory=FieldResult, description="Nationality code")
    passport_number: FieldResult = Field(default_factory=FieldResult, description="Passport number")
    passport_expiry_date: FieldResult = Field(default_factory=FieldResult, description="Expiration date (YYYY-MM-DD)")
    valid_mrz: bool = Field(default=False, description="True if MRZ checksum passed")


class VisaDocumentRecord(BaseModel):
    """
    Isolated extraction attributes from full e-Visa paper / PDF printout.
    """
    visa_number: FieldResult = Field(default_factory=FieldResult, description="e-Visa number")
    visa_type: FieldResult = Field(default_factory=FieldResult, description="e-Visa category")
    visa_issue_date: FieldResult = Field(default_factory=FieldResult, description="Issue date (YYYY-MM-DD)")
    visa_expiry_date: FieldResult = Field(default_factory=FieldResult, description="Expiry date (YYYY-MM-DD)")
    visa_place_of_issue: FieldResult = Field(default_factory=FieldResult, description="Authority / mission")


class VisaStampRecord(BaseModel):
    """
    Isolated extraction attributes from passport booklet visa ink stamp / vignette.
    """
    visa_number: FieldResult = Field(default_factory=FieldResult, description="Stamp visa number")
    visa_type: FieldResult = Field(default_factory=FieldResult, description="Stamp visa category")
    visa_issue_date: FieldResult = Field(default_factory=FieldResult, description="Issue date (YYYY-MM-DD)")
    visa_expiry_date: FieldResult = Field(default_factory=FieldResult, description="Expiry date (YYYY-MM-DD)")
    visa_place_of_issue: FieldResult = Field(default_factory=FieldResult, description="Issuing mission")


class EntryStampRecord(BaseModel):
    """
    Isolated extraction attributes from immigration rubber arrival stamp.
    """
    arrival_date_india: FieldResult = Field(default_factory=FieldResult, description="Arrival date (YYYY-MM-DD)")
    arrival_port_india: FieldResult = Field(default_factory=FieldResult, description="Port or airport of arrival")


class FormCReportRecord(BaseModel):
    """
    Isolated extraction attributes from Form-C Rule 14 handwritten arrival report.
    """
    name_of_foreigner: FieldResult = Field(default_factory=FieldResult, description="Row 1: Name")
    permanent_address: FieldResult = Field(default_factory=FieldResult, description="Row 2: Street address")
    city: FieldResult = Field(default_factory=FieldResult, description="Row 2: City")
    country: FieldResult = Field(default_factory=FieldResult, description="Row 2: Country")
    contact_email: FieldResult = Field(default_factory=FieldResult, description="Row 2: Email")
    contact_phone: FieldResult = Field(default_factory=FieldResult, description="Row 2: Mobile / Phone")
    nationality: FieldResult = Field(default_factory=FieldResult, description="Row 3: Nationality")
    place_of_issue_of_visa: FieldResult = Field(default_factory=FieldResult, description="Row 4: Visa place")
    date_of_arrival_in_india: FieldResult = Field(default_factory=FieldResult, description="Row 5: Arrival date")
    arrival_datetime_homestay: FieldResult = Field(default_factory=FieldResult, description="Row 6: Homestay arrival")
    arrived_from: FieldResult = Field(default_factory=FieldResult, description="Row 7: Arrived from")
    proceeding_to: FieldResult = Field(default_factory=FieldResult, description="Row 8: Proceeding to")
    stay_duration_india: FieldResult = Field(default_factory=FieldResult, description="Row 9: Stay duration")
    purpose_of_visit: FieldResult = Field(default_factory=FieldResult, description="Row 10: Purpose & profession")
    expected_departure_date: FieldResult = Field(default_factory=FieldResult, description="Row 11: Departure date")
    arrived_country_city_india: FieldResult = Field(default_factory=FieldResult, description="Row 12: Country & city")


class DocumentExtractionContainer(BaseModel):
    """
    Encapsulates each document's isolated extraction output.
    Ensures that values from one document never overwrite or conflict with another.
    """
    passport_bio: Optional[PassportBioRecord] = Field(
        default=None,
        description="Isolated attributes from Passport Bio-Data page"
    )
    visa_document: Optional[VisaDocumentRecord] = Field(
        default=None,
        description="Isolated attributes from e-Visa paper document"
    )
    visa_stamp: Optional[VisaStampRecord] = Field(
        default=None,
        description="Isolated attributes from passport booklet visa stamp"
    )
    entry_stamp: Optional[EntryStampRecord] = Field(
        default=None,
        description="Isolated attributes from immigration arrival stamp"
    )
    welcome_form: Optional[FormCReportRecord] = Field(
        default=None,
        description="Isolated attributes from Form-C handwritten report"
    )


# ==============================================================================
# Fused Form-C Model (Backwards Compatible)
# ==============================================================================

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

    # --- Isolated Per-Document Data Bucket (Task 45) ---
    documents: Optional[DocumentExtractionContainer] = Field(
        default_factory=DocumentExtractionContainer,
        description="Isolated per-document raw extraction results"
    )

    # --- Pipeline Diagnostics ---
    processing_notes: List[str] = Field(
        default_factory=list,
        description="Diagnostic notes or confidence comparison logs across extraction stages"
    )


# ==============================================================================
# Job API Request & Response Schemas
# ==============================================================================

class JobSubmissionResponse(BaseModel):
    """
    Immediate acknowledgement response returned when document images are ingested.
    Provides tracking UUID for asynchronous status polling.
    """
    job_id: str = Field(
        ...,
        description="Unique UUID tracking identifier for the background processing task"
    )
    status: JobStatus = Field(
        default=JobStatus.QUEUED,
        description="Current state of the asynchronous pipeline"
    )
    created_at: str = Field(
        ...,
        description="ISO 8601 formatted timestamp of job intake"
    )
    message: str = Field(
        default="Documents queued for extraction. Poll /api/v1/jobs/{job_id} for completion.",
        description="Human-readable status notice"
    )


class JobDetailResponse(BaseModel):
    """
    Detailed status and result payload returned upon polling /api/v1/jobs/{job_id}.
    """
    job_id: str = Field(..., description="UUID tracking identifier")
    status: JobStatus = Field(..., description="Current job status")
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    updated_at: Optional[str] = Field(default=None, description="ISO 8601 last modified timestamp")
    result: Optional[GuestFormCRecord] = Field(
        default=None,
        description="Unified Form-C guest record with field-level confidence and per-document buckets"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Details of unhandled execution failure if processing aborted"
    )


class JobDeletionResponse(BaseModel):
    """
    Confirmation response returned when a submission is deleted (Task 44).
    """
    job_id: str = Field(..., description="UUID of deleted job")
    status: str = Field(default="DELETED", description="Deletion outcome status")
    purged_disk_files: int = Field(default=0, description="Count of persisted disk images removed")
    message: str = Field(default="Job record and stored image artifacts successfully deleted.", description="Status note")