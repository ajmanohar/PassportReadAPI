"""
==============================================================================
routers/jobs_router.py - PassportReadAPI
==============================================================================
Modular API router for asynchronous Form-C batch extraction jobs.
Provides non-blocking intake endpoints for multi-document bundles,
supports granular supplementary document attachments, automatic background
worker dispatching, diagnostic polling routes, binary image streaming
for desktop and mobile document review, and cascading job deletion.

Endpoints:
  - POST   /api/v1/jobs/submit              : Initial intake with passport_bio & optional docs
  - POST   /api/v1/jobs/{job_id}/attach/{t} : Granular secondary document upload
  - POST   /api/v1/jobs/{job_id}/start      : Safe trigger to enqueue pending jobs
  - GET    /api/v1/jobs/{job_id}/image/{t}  : Raw binary JPEG image stream
  - GET    /api/v1/jobs/{job_id}            : Polling status and composite extraction
  - GET    /api/v1/jobs                     : Chronological recent jobs listing
  - DELETE /api/v1/jobs/{job_id}            : Cascade purge of database record & disk images
==============================================================================
"""

import json
import logging
import os
import shutil
import uuid
from typing import List, Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import ValidationError

# Configuration, database, and schema imports
from config import settings
from storage.db import (
    create_job,
    get_job,
    list_recent_jobs,
    get_db_connection,
    delete_job_record
)
from schemas.job import (
    JobSubmissionResponse,
    JobDetailResponse,
    JobDeletionResponse,
    JobStatus,
    GuestFormCRecord
)
from services.job_worker import safe_enqueue_job

# Configure router logger for diagnostic tracing
logger = logging.getLogger("jobs_router")
logger.setLevel(logging.INFO)

# ==============================================================================
# Router Configuration & Storage Paths
# ==============================================================================

# Initialize APIRouter with standard Form-C batch prefix and tags
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

    :param upload_file: FastAPI UploadFile handle received from multipart request.
    :param destination_path: Absolute target path on disk where bytes will be written.
    """
    with open(destination_path, "wb") as buffer:
        # Stream file in 1 MB chunks to prevent high memory spikes during upload
        while content := await upload_file.read(1024 * 1024):
            buffer.write(content)


# ==============================================================================
# Job Intake Endpoint
# ==============================================================================

@router.post(
    "/submit",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Initialize Form-C job and ingest initial passport bio page"
)
async def submit_documents_job(
    passport_bio: UploadFile = File(..., description="Passport identity page image containing photo and MRZ lines"),
    visa_document: Optional[UploadFile] = File(None, description="Optional image/PDF of e-Visa paper document"),
    visa_stamp: Optional[UploadFile] = File(None, description="Optional image of Passport booklet visa ink stamp / vignette"),
    visa_page: Optional[UploadFile] = File(None, description="Legacy alias: image of Indian Visa sticker or document page"),
    entry_stamp: Optional[UploadFile] = File(None, description="Optional image of Immigration Arrival rubber ink stamp"),
    welcome_form: Optional[UploadFile] = File(None, description="Optional image of physical handwritten Form-C Arrival Report")
) -> JobSubmissionResponse:
    """
    Initializes a new job directory, stores the mandatory passport bio page and any bundled documents,
    creates a QUEUED record in the SQLite database, and safely enqueues the worker with deduplication.
    """
    # 1. Generate unique job identifier (UUID4) and dedicated disk directory
    job_id = str(uuid.uuid4())
    job_dir = os.path.join(BASE_STORAGE_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    # Dictionary to keep track of stored document paths on disk
    doc_paths = {}

    try:
        # 2. Persist mandatory Passport Bio image
        bio_filename = f"bio_{passport_bio.filename}"
        bio_path = os.path.join(job_dir, bio_filename)
        await save_uploaded_file(passport_bio, bio_path)
        doc_paths["passport_bio"] = bio_path

        # 3. Persist optional e-Visa Paper Document if supplied
        if visa_document and visa_document.filename:
            vdoc_filename = f"visadoc_{visa_document.filename}"
            vdoc_path = os.path.join(job_dir, vdoc_filename)
            await save_uploaded_file(visa_document, vdoc_path)
            doc_paths["visa_document"] = vdoc_path

        # 4. Persist optional Passport Visa Stamp if supplied
        if visa_stamp and visa_stamp.filename:
            vstamp_filename = f"visastamp_{visa_stamp.filename}"
            vstamp_path = os.path.join(job_dir, vstamp_filename)
            await save_uploaded_file(visa_stamp, vstamp_path)
            doc_paths["visa_stamp"] = vstamp_path

        # 5. Legacy backward compatibility: persist visa_page if provided
        if visa_page and visa_page.filename and "visa_document" not in doc_paths and "visa_stamp" not in doc_paths:
            visa_filename = f"visa_{visa_page.filename}"
            visa_path = os.path.join(job_dir, visa_filename)
            await save_uploaded_file(visa_page, visa_path)
            doc_paths["visa_page"] = visa_path

        # 6. Persist optional Entry Stamp image if supplied upfront
        if entry_stamp and entry_stamp.filename:
            stamp_filename = f"stamp_{entry_stamp.filename}"
            stamp_path = os.path.join(job_dir, stamp_filename)
            await save_uploaded_file(entry_stamp, stamp_path)
            doc_paths["entry_stamp"] = stamp_path

        # 7. Persist optional Handwritten Welcome Form image if supplied upfront
        if welcome_form and welcome_form.filename:
            form_filename = f"form_{welcome_form.filename}"
            form_path = os.path.join(job_dir, form_filename)
            await save_uploaded_file(welcome_form, form_path)
            doc_paths["welcome_form"] = form_path

    except Exception as exc:
        # Clean up disk directory on failure to avoid orphaned partial uploads
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to persist uploaded document files: {str(exc)}"
        )

    # 8. Insert initial QUEUED entry into SQLite database
    job_meta = create_job(job_id, doc_paths)

    # 9. Safely enqueue into background worker queue with duplicate guard
    await safe_enqueue_job(job_id)
    logger.info(f"[*] Ingested job {job_id} and dispatched to processing queue safely.")

    return JobSubmissionResponse(
        job_id=job_id,
        status=JobStatus.QUEUED,
        created_at=job_meta["created_at"]
    )


# ==============================================================================
# Granular Attachment Endpoints
# ==============================================================================

@router.post(
    "/{job_id}/attach/{doc_type}",
    summary="Attach an additional document image to an existing job"
)
async def attach_document_to_job(
    job_id: str,
    doc_type: str,
    file: UploadFile = File(...)
):
    """
    Attaches secondary captures (visa_document, visa_stamp, visa_page, entry_stamp, welcome_form, guest_photo)
    to an existing job directory and updates the database path manifest.
    """
    valid_types = {
        "visa_document",
        "visa_stamp",
        "visa_page",
        "entry_stamp",
        "welcome_form",
        "guest_photo"
    }
    if doc_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid doc_type '{doc_type}'. Expected one of: {list(valid_types)}"
        )

    # Verify that the parent job exists in SQLite
    job_record = get_job(job_id)
    if not job_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found."
        )

    job_dir = os.path.join(BASE_STORAGE_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    dest_filename = f"{doc_type}_{file.filename}"
    dest_path = os.path.join(job_dir, dest_filename)
    await save_uploaded_file(file, dest_path)

    # Update database record doc_paths manifest
    doc_paths = job_record.get("doc_paths", {})
    doc_paths[doc_type] = dest_path

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE jobs SET doc_paths_json = ? WHERE job_id = ?;",
            (json.dumps(doc_paths), job_id)
        )
        conn.commit()

    logger.info(f"[*] Attached {doc_type} to job {job_id} (path: {dest_path}).")
    return {"status": "success", "job_id": job_id, "attached": doc_type}


# ==============================================================================
# Pipeline Execution Trigger Endpoint
# ==============================================================================

@router.post(
    "/{job_id}/start",
    summary="Ensure background batch extraction is queued and running"
)
async def start_job_processing(job_id: str):
    """
    Trigger to safely enqueue a job for worker processing.
    Uses safe_enqueue_job() to strictly guard against double-queuing while
    a job is already waiting in queue, actively processing, or already completed.
    """
    job_record = get_job(job_id)
    if not job_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found."
        )

    current_status = job_record.get("status")

    # Safe deduplicated enqueue
    enqueued = await safe_enqueue_job(job_id)

    if not enqueued:
        return {
            "status": current_status,
            "job_id": job_id,
            "message": f"Job {job_id} is already in state '{current_status}' or already queued; duplicate dispatch skipped."
        }

    return {
        "status": "QUEUED",
        "job_id": job_id,
        "message": f"Job {job_id} successfully queued for background extraction."
    }


# ==============================================================================
# Image Streaming Endpoint (Desktop & Mobile Document Review)
# ==============================================================================

@router.get(
    "/{job_id}/image/{doc_type}",
    summary="Retrieve captured document image artifact for desktop review",
    description="Streams the raw image file (passport_bio, visa_document, visa_stamp, visa_page, entry_stamp, welcome_form, or guest_photo) stored on disk for a specific job."
)
async def get_job_document_image(job_id: str, doc_type: str):
    """
    Streams the stored binary document image for desktop/mobile inspection.
    First checks SQLite doc_paths manifest, then falls back to directory inspection.
    """
    job_record = get_job(job_id)
    if not job_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found."
        )

    doc_paths = job_record.get("doc_paths", {})
    resolved_path = doc_paths.get(doc_type)

    if resolved_path and os.path.exists(resolved_path):
        return FileResponse(resolved_path, media_type="image/jpeg")

    job_dir = os.path.join(BASE_STORAGE_DIR, job_id)
    if os.path.exists(job_dir):
        prefix_mapping = {
            "passport_bio": ["bio_", "passport_"],
            "visa_document": ["visadoc_"],
            "visa_stamp": ["visastamp_"],
            "visa_page": ["visa_", "visadoc_", "visastamp_"],
            "entry_stamp": ["stamp_"],
            "welcome_form": ["form_", "welcome_"],
            "guest_photo": ["guest_photo_", "photo_"]
        }
        prefixes = prefix_mapping.get(doc_type, [f"{doc_type}_"])
        for fname in os.listdir(job_dir):
            for prefix in prefixes:
                if fname.lower().startswith(prefix):
                    candidate = os.path.join(job_dir, fname)
                    if os.path.isfile(candidate):
                        return FileResponse(candidate, media_type="image/jpeg")

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Document image '{doc_type}' not found for job ID: {job_id}"
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
    When COMPLETED, returns the fused Form-C guest record with field-level confidence ratings
    and per-document isolated extraction models.
    """
    job_record = get_job(job_id)
    if not job_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found."
        )

    result_record: Optional[GuestFormCRecord] = None
    if job_record.get("result"):
        try:
            result_record = GuestFormCRecord(**job_record["result"])
        except (ValidationError, TypeError, Exception) as parse_err:
            logger.warning(f"Could not strictly validate result for job {job_id}: {parse_err}")
            result_record = None

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
        result_record: Optional[GuestFormCRecord] = None
        if item.get("result"):
            try:
                result_record = GuestFormCRecord(**item["result"])
            except Exception as parse_err:
                logger.warning(f"Error parsing job {item.get('job_id')} result: {parse_err}")
                result_record = None

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


# ==============================================================================
# Job Deletion Endpoint (Task 44: Mobile-Triggered Deletion)
# ==============================================================================

@router.delete(
    "/{job_id}",
    response_model=JobDeletionResponse,
    summary="Permanently delete job record and wipe associated image artifacts from disk"
)
async def delete_job_and_artifacts(job_id: str) -> JobDeletionResponse:
    """
    Permanently purges a submission:
    1. Removes the job record from SQLite database.
    2. Deletes the entire disk folder /opt/PassportReadAPI/storage/jobs/{job_id} and all stored images.
    """
    job_record = get_job(job_id)
    if not job_record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found."
        )

    purged_files = 0
    job_dir = os.path.join(BASE_STORAGE_DIR, job_id)
    if os.path.exists(job_dir):
        try:
            purged_files = len([f for f in os.listdir(job_dir) if os.path.isfile(os.path.join(job_dir, f))])
            shutil.rmtree(job_dir, ignore_errors=True)
            logger.info(f"[*] Purged disk directory and {purged_files} file(s) for job: {job_id}")
        except Exception as disk_err:
            logger.warning(f"[*] Error removing disk files for job {job_id}: {disk_err}")

    delete_job_record(job_id)
    logger.info(f"[+] Successfully purged job {job_id} from database and disk storage.")

    return JobDeletionResponse(
        job_id=job_id,
        status="DELETED",
        purged_disk_files=purged_files,
        message=f"Submission {job_id} and {purged_files} stored image artifact(s) permanently deleted."
    )