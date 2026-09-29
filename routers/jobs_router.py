"""
Modular API router for asynchronous Form-C batch extraction jobs.
Provides non-blocking intake endpoints for multi-document bundles,
enqueuing background extraction tasks and exposing status polling routes.
"""

import os
import shutil
import uuid
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, status

# Configuration, database, and schema imports
from config import settings
from storage.db import create_job, get_job, list_recent_jobs
from schemas.job import (
    JobSubmissionResponse,
    JobDetailResponse,
    JobStatus,
    GuestFormCRecord
)
from services.job_worker import job_queue

# ==============================================================================
# Router Configuration & Storage Paths
# ==============================================================================

router = APIRouter(
    prefix="/jobs",
    tags=["Asynchronous Batch Extraction"]
)

# Base disk directory for persisting uploaded document images during asynchronous processing
BASE_STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "jobs")
os.makedirs(BASE_STORAGE_DIR, exist_ok=True)


# ==============================================================================
# Helper Function for File Saving
# ==============================================================================

async def save_uploaded_file(upload_file: UploadFile, destination_path: str) -> None:
    """
    Asynchronously streams uploaded multipart file bytes into a local filesystem path.

    :param upload_file: FastAPI UploadFile handle.
    :param destination_path: Target disk path.
    """
    with open(destination_path, "wb") as buffer:
        while content := await upload_file.read(1024 * 1024):  # Read in 1MB chunks
            buffer.write(content)


# ==============================================================================
# Job Intake Endpoint (Immediate 202 Accepted)
# ==============================================================================

@router.post(
    "/submit",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit multi-document bundle for asynchronous Form-C processing"
)
async def submit_documents_job(
    passport_bio: UploadFile = File(..., description="Passport identity page image containing photo and MRZ lines"),
    visa_page: Optional[UploadFile] = File(None, description="Image of Indian Visa sticker or OCI document page"),
    entry_stamp: Optional[UploadFile] = File(None, description="Image of Immigration Arrival / Entry rubber ink stamp"),
    welcome_form: Optional[UploadFile] = File(None, description="Image of physical handwritten Form-C Arrival Report")
) -> JobSubmissionResponse:
    """
    Accepts up to 4 guest document images, stores them on disk, creates a QUEUED database record,
    pushes the task to the background processing queue, and immediately returns a job tracking ID.
    """
    # 1. Generate unique job identifier and dedicated disk directory
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(BASE_STORAGE_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    doc_paths = {}

    try:
        # 2. Persist mandatory Passport Bio image
        bio_filename = f"bio_{passport_bio.filename}"
        bio_path = os.path.join(job_dir, bio_filename)
        await save_uploaded_file(passport_bio, bio_path)
        doc_paths["passport_bio"] = bio_path

        # 3. Persist optional Visa / OCI image if supplied
        if visa_page and visa_page.filename:
            visa_filename = f"visa_{visa_page.filename}"
            visa_path = os.path.join(job_dir, visa_filename)
            await save_uploaded_file(visa_page, visa_path)
            doc_paths["visa_page"] = visa_path

        # 4. Persist optional Entry Stamp image if supplied
        if entry_stamp and entry_stamp.filename:
            stamp_filename = f"stamp_{entry_stamp.filename}"
            stamp_path = os.path.join(job_dir, stamp_filename)
            await save_uploaded_file(entry_stamp, stamp_path)
            doc_paths["entry_stamp"] = stamp_path

        # 5. Persist optional Handwritten Welcome Form image if supplied
        if welcome_form and welcome_form.filename:
            form_filename = f"form_{welcome_form.filename}"
            form_path = os.path.join(job_dir, form_filename)
            await save_uploaded_file(welcome_form, form_path)
            doc_paths["welcome_form"] = form_path

    except Exception as exc:
        # Clean up storage on upload failure
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist uploaded document files: {str(exc)}"
        )

    # 6. Insert QUEUED entry into SQLite database
    job_meta = create_job(job_id, doc_paths)

    # 7. Push job identifier to the serial background queue
    await job_queue.put(job_id)

    return JobSubmissionResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        created_at=job_meta["created_at"]
    )


# ==============================================================================
# Job Polling Endpoint
# ==============================================================================

@router.get(
    "/{job_id}",
    response_model=JobDetailResponse,
    summary="Poll job processing status and retrieve fused Form-C extraction"
)
async def get_job_status(job_id: str) -> JobDetailResponse:
    """
    Returns current execution state (QUEUED, PROCESSING, COMPLETED, FAILED).
    When COMPLETED, returns the fused Form-C guest record.
    """
    job_record = get_job(job_id)
    if not job_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found."
        )

    # Map raw dictionary into GuestFormCRecord if complete
    result_record = None
    if job_record.get("result"):
        result_record = GuestFormCRecord(**job_record["result"])

    return JobDetailResponse(
        job_id=job_record["job_id"],
        status=JobStatus(job_record["status"]),
        created_at=job_record["created_at"],
        updated_at=job_record["updated_at"],
        result=result_record,
        error_message=job_record["error_message"]
    )


# ==============================================================================
# Job Sync / Listing Endpoint
# ==============================================================================

@router.get(
    "",
    response_model=List[JobDetailResponse],
    summary="List recent jobs for Android Room DB synchronization"
)
async def list_jobs(limit: int = 50) -> List[JobDetailResponse]:
    """
    Returns the most recent jobs ordered descending by submission time.
    """
    jobs = list_recent_jobs(limit=limit)
    response_items = []
    for item in jobs:
        result_record = None
        if item.get("result"):
            result_record = GuestFormCRecord(**item["result"])

        response_items.append(
            JobDetailResponse(
                job_id=item["job_id"],
                status=JobStatus(item["status"]),
                created_at=item["created_at"],
                updated_at=item["updated_at"],
                result=result_record,
                error_message=item["error_message"]
            )
        )
    return response_items