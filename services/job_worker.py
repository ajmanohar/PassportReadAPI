"""
Background batch execution pipeline and serial queue worker for asynchronous Form-C jobs.
Processes multi-document bundles (Passport Bio, Visa, Entry Stamp, Welcome Form) sequentially
using Tesseract OCR for checksum validation and Qwen2-VL for high-resolution visual extraction.
Persists progress and results directly to the SQLite database.
"""

import asyncio
import io
import json
import logging
import os
import re
from typing import Dict, Any, Optional

from PIL import Image
import torch
from passporteye import read_mrz

# Central configuration and repository imports
from config import settings
from storage.db import update_job_status, save_job_result, get_job
from schemas.job import GuestFormCRecord

# Set up module-level logger
logger = logging.getLogger("job_worker")
logger.setLevel(logging.INFO)

# Global in-memory queue holding job UUID strings for sequential execution
job_queue: asyncio.Queue = asyncio.Queue()


# ==============================================================================
# Image Preparation Helper
# ==============================================================================

def load_and_preprocess_image(file_path: str, max_dimension: int = 1280) -> Optional[Image.Image]:
    """
    Loads an image from local disk storage, standardizes color channels to RGB,
    and downscales proportionally if the maximum dimension exceeds configured limits.

    :param file_path: Absolute or relative filesystem path to the image file.
    :param max_dimension: Maximum allowed width or height in pixels.
    :return: Prepared PIL Image instance, or None if the file cannot be read.
    """
    if not os.path.exists(file_path):
        logger.warning(f"File not found on disk: {file_path}")
        return None

    try:
        image = Image.open(file_path)

        # Standardize color profile to 3-channel RGB (removes alpha or CMYK artifacts)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Downscale proportionally to avoid memory bloat while preserving sharp text
        if max(image.size) > max_dimension:
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)

        return image
    except Exception as exc:
        logger.error(f"Failed to load or preprocess image from '{file_path}': {str(exc)}")
        return None


# ==============================================================================
# Inference Subroutine Helpers
# ==============================================================================

def run_vlm_inference(image: Image.Image, system_prompt: str, ml_state: dict) -> dict:
    """
    Executes a structured query against the shared in-memory Qwen2-VL model instance
    and parses the output into a Python dictionary.

    :param image: Preprocessed PIL Image instance.
    :param system_prompt: Guidance text instructing the VLM on expected JSON fields.
    :param ml_state: Shared dictionary holding 'model' and 'processor' instances.
    :return: Extracted key-value dictionary parsed from the model's generated JSON.
    """
    from qwen_vl_utils import process_vision_info

    processor = ml_state.get("processor")
    model = ml_state.get("model")

    if not model or not processor:
        raise RuntimeError("Qwen2-VL model and processor are not loaded in ml_state.")

    # Format user prompt according to Qwen2-VL chat conversation template
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    # Bound visual tokens for clean inference without memory blowouts
                    "min_pixels": settings.min_pixels,
                    "max_pixels": settings.max_pixels
                },
                {"type": "text", "text": system_prompt}
            ]
        }
    ]

    # Process prompt template and extract visual input tensors
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    # Build input tensor dictionaries
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    # Move tensors to designated execution device (CPU)
    inputs = {k: v.to(settings.device) for k, v in inputs.items()}

    # Run deterministic inference without calculating gradients
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            do_sample=False
        )

    # Separate model-generated response tokens from the input prompt tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    # Decode tokens into UTF-8 string
    raw_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    # Extract JSON string payload bounded by braces
    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(raw_text)
    except Exception:
        logger.warning(f"Failed to parse JSON from VLM output. Raw text: {raw_text[:200]}")
        return {"confidence_note": f"Raw model response: {raw_text}"}


# ==============================================================================
# Job Processing Pipeline
# ==============================================================================

async def process_single_job(job_id: str, ml_state: dict) -> None:
    """
    Executes the multi-document extraction pipeline for a single queued job.
    Runs Tesseract MRZ, visual bio fallback, Visa parsing, Stamp extraction,
    and handwritten form OCR sequentially, then fuses all fields.

    :param job_id: Unique UUID string identifier of the target job.
    :param ml_state: Shared dictionary holding model handles.
    """
    logger.info(f"[*] Starting processing for job: {job_id}")
    update_job_status(job_id, "PROCESSING")

    # Fetch document file paths recorded during job intake
    job_record = get_job(job_id)
    if not job_record:
        logger.error(f"Job record {job_id} not found in database.")
        return

    doc_paths: Dict[str, str] = job_record.get("doc_paths", {})
    fused_record = GuestFormCRecord()
    processing_notes = []

    try:
        # ----------------------------------------------------------------------
        # Stage 1: Tesseract MRZ Checksum Extraction (Passport Bio Page)
        # ----------------------------------------------------------------------
        bio_path = doc_paths.get("passport_bio")
        mrz_success = False

        if bio_path and os.path.exists(bio_path):
            logger.info(f"[{job_id}] Stage 1: Running Tesseract MRZ checksum extraction...")
            try:
                # Read binary bytes from disk and parse via PassportEye in worker thread
                with open(bio_path, "rb") as f:
                    image_bytes = f.read()

                mrz_record = await asyncio.to_thread(read_mrz, io.BytesIO(image_bytes))

                if mrz_record is not None:
                    mrz_data = mrz_record.to_dict() if hasattr(mrz_record, "to_dict") else {}
                    score = getattr(mrz_record, "valid_score", 0)
                    is_valid = bool(score >= 80) if isinstance(score, (int, float)) else bool(score)

                    fused_record.valid_mrz = is_valid
                    fused_record.surname = mrz_data.get("surname")
                    fused_record.given_names = mrz_data.get("names")
                    fused_record.passport_number = mrz_data.get("number")
                    fused_record.nationality = mrz_data.get("nationality")
                    fused_record.date_of_birth = mrz_data.get("date_of_birth")
                    fused_record.sex = mrz_data.get("sex")
                    fused_record.passport_expiry_date = mrz_data.get("expiration_date")
                    mrz_success = True
                    processing_notes.append(f"MRZ parsed successfully (checksum valid: {is_valid}).")
                else:
                    processing_notes.append("MRZ not detected; deferring to visual bio extraction.")
            except Exception as exc:
                logger.warning(f"[{job_id}] MRZ extraction exception: {str(exc)}")
                processing_notes.append(f"MRZ extraction warning: {str(exc)}")

        # ----------------------------------------------------------------------
        # Stage 2: Qwen2-VL Visual Bio Page Extraction (Fallback / Verification)
        # ----------------------------------------------------------------------
        if bio_path and (not mrz_success or not fused_record.surname or not fused_record.passport_number):
            logger.info(f"[{job_id}] Stage 2: Running Qwen2-VL visual bio-page extraction...")
            bio_image = await asyncio.to_thread(load_and_preprocess_image, bio_path)
            if bio_image:
                bio_prompt = (
                    "Extract passport bio data from the image. Output valid JSON with keys: "
                    "'surname', 'given_names', 'passport_number', 'nationality', "
                    "'date_of_birth' (YYYY-MM-DD), 'sex' ('M'/'F'), 'expiry_date' (YYYY-MM-DD), 'confidence_note'."
                )
                vlm_bio = await asyncio.to_thread(run_vlm_inference, bio_image, bio_prompt, ml_state)

                # Backfill missing fields from VLM
                fused_record.surname = fused_record.surname or vlm_bio.get("surname")
                fused_record.given_names = fused_record.given_names or vlm_bio.get("given_names")
                fused_record.passport_number = fused_record.passport_number or vlm_bio.get("passport_number")
                fused_record.nationality = fused_record.nationality or vlm_bio.get("nationality")
                fused_record.date_of_birth = fused_record.date_of_birth or vlm_bio.get("date_of_birth")
                fused_record.sex = fused_record.sex or vlm_bio.get("sex")
                fused_record.passport_expiry_date = fused_record.passport_expiry_date or vlm_bio.get("expiry_date")
                processing_notes.append("Visual bio extraction executed via Qwen2-VL.")

        # ----------------------------------------------------------------------
        # Stage 3: Qwen2-VL Visa / OCI Extraction
        # ----------------------------------------------------------------------
        visa_path = doc_paths.get("visa_page")
        if visa_path and os.path.exists(visa_path):
            logger.info(f"[{job_id}] Stage 3: Running Qwen2-VL visa extraction...")
            visa_image = await asyncio.to_thread(load_and_preprocess_image, visa_path)
            if visa_image:
                visa_prompt = (
                    "Extract visa details from the image. Output valid JSON with keys: "
                    "'visa_number', 'visa_type', 'issue_date' (YYYY-MM-DD), 'expiry_date' (YYYY-MM-DD), "
                    "'place_of_issue', 'confidence_note'."
                )
                vlm_visa = await asyncio.to_thread(run_vlm_inference, visa_image, visa_prompt, ml_state)
                fused_record.visa_number = vlm_visa.get("visa_number")
                fused_record.visa_type = vlm_visa.get("visa_type")
                fused_record.visa_issue_date = vlm_visa.get("issue_date")
                fused_record.visa_expiry_date = vlm_visa.get("expiry_date")
                fused_record.visa_place_of_issue = vlm_visa.get("place_of_issue")
                processing_notes.append("Visa / OCI details extracted via Qwen2-VL.")

        # ----------------------------------------------------------------------
        # Stage 4: Qwen2-VL Immigration Entry Stamp Extraction
        # ----------------------------------------------------------------------
        stamp_path = doc_paths.get("entry_stamp")
        if stamp_path and os.path.exists(stamp_path):
            logger.info(f"[{job_id}] Stage 4: Running Qwen2-VL entry stamp extraction...")
            stamp_image = await asyncio.to_thread(load_and_preprocess_image, stamp_path)
            if stamp_image:
                stamp_prompt = (
                    "Extract arrival stamp details from the image. Output valid JSON with keys: "
                    "'arrival_date' (YYYY-MM-DD), 'arrival_port', 'confidence_note'."
                )
                vlm_stamp = await asyncio.to_thread(run_vlm_inference, stamp_image, stamp_prompt, ml_state)
                fused_record.arrival_date_india = vlm_stamp.get("arrival_date")
                fused_record.arrival_port_india = vlm_stamp.get("arrival_port")
                processing_notes.append("Entry stamp details extracted via Qwen2-VL.")

        # ----------------------------------------------------------------------
        # Stage 5: Qwen2-VL Handwritten Welcome Form Extraction (Form-C Rule 14)
        # ----------------------------------------------------------------------
        welcome_path = doc_paths.get("welcome_form")
        if welcome_path and os.path.exists(welcome_path):
            logger.info(f"[{job_id}] Stage 5: Running Qwen2-VL handwritten welcome form extraction...")
            welcome_image = await asyncio.to_thread(load_and_preprocess_image, welcome_path)
            if welcome_image:
                welcome_prompt = (
                    "Extract handwritten Form C Arrival Report fields from the image. Output valid JSON with keys: "
                    "'full_address', 'email', 'mobile', 'arrived_from', 'proceeding_to', "
                    "'purpose_of_visit_profession', 'confidence_note'."
                )
                vlm_form = await asyncio.to_thread(run_vlm_inference, welcome_image, welcome_prompt, ml_state)
                fused_record.permanent_address = vlm_form.get("full_address")
                fused_record.contact_email = vlm_form.get("email")
                fused_record.contact_phone = vlm_form.get("mobile")
                fused_record.arrived_from = vlm_form.get("arrived_from")
                fused_record.proceeding_to = vlm_form.get("proceeding_to")
                fused_record.purpose_of_visit = vlm_form.get("purpose_of_visit_profession")
                processing_notes.append("Handwritten welcome form extracted via Qwen2-VL.")

        # Record diagnostic notes and persist final fused record
        fused_record.processing_notes = processing_notes
        save_job_result(job_id, fused_record.model_dump())
        logger.info(f"[+] Job {job_id} successfully processed and marked COMPLETED.")

    except Exception as exc:
        logger.error(f"[!] Processing failed for job {job_id}: {str(exc)}", exc_info=True)
        update_job_status(job_id, "FAILED", error_message=str(exc))


# ==============================================================================
# Asynchronous Background Worker Loop
# ==============================================================================

async def background_queue_worker(ml_state: dict) -> None:
    """
    Continuous background loop that waits for jobs added to job_queue
    and processes them sequentially one after another.

    :param ml_state: Global dictionary containing warm Qwen2-VL model handles.
    """
    logger.info("[*] Background async queue worker loop initialized.")
    while True:
        try:
            # Await the next job UUID from the queue
            job_id = await job_queue.get()
            logger.info(f"[*] Queue dispatched job: {job_id}")

            # Execute pipeline
            await process_single_job(job_id, ml_state)

            # Signal that the current item has completed processing
            job_queue.task_done()
        except asyncio.CancelledError:
            logger.info("[*] Background async queue worker task was cancelled.")
            break
        except Exception as exc:
            logger.error(f"[!] Unexpected error in queue worker loop: {str(exc)}", exc_info=True)
            await asyncio.sleep(1.0)