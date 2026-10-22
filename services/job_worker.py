# ==============================================================================
# job_worker.py - PassportReadAPI
# ==============================================================================
# Background batch execution pipeline and serial queue worker for asynchronous Form-C jobs.
# Processes multi-document bundles sequentially:
#   Stage 1: Passport Bio-Data (4-Angle MRZ Tournament + Upright Qwen2-VL) [Task 50]
#   Stage 2A: e-Visa Paper / PDF Document Extraction (Qwen2-VL) [Task 42]
#   Stage 2B: Passport Booklet Consular Visa Stamp Extraction (Qwen2-VL) [Task 42]
#   Stage 3: Immigration Arrival Rubber Ink Stamp Extraction (Qwen2-VL with multi-angle guard) [Task 49]
#   Stage 4: Handwritten Form-C Rule 14 Arrival Report (OCR-grounded full-page Qwen2-VL)
#
# Key Architectural Features:
# - Task 45: Populates isolated per-document data models in fused_record.documents.
# - Task 50: 4-way rotation scan for Passport Bio page to maximize MRZ checksum passing.
# - Loop/Duplicate Guard: Enforces safe deduplicated queuing and skips re-execution
#   if a job is already COMPLETED or already queued/active.
# ==============================================================================

import asyncio
import io
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Set

from PIL import Image, ImageOps, ImageEnhance
import pytesseract
import torch
from passporteye import read_mrz

# Central configuration and repository imports
from config import settings
from storage.db import update_job_status, save_intermediate_job_result, save_job_result, get_job
from schemas.job import (
    GuestFormCRecord,
    FieldResult,
    DocumentExtractionContainer,
    PassportBioRecord,
    VisaDocumentRecord,
    VisaStampRecord,
    EntryStampRecord,
    FormCReportRecord
)

# Set up module-level logger
logger = logging.getLogger("job_worker")
logger.setLevel(logging.INFO)

# Global in-memory queue holding job tracking identifiers for sequential execution by the worker loop
job_queue: asyncio.Queue = asyncio.Queue()

# In-memory tracking set to guard against duplicate queuing while a job is pending or active
enqueued_or_active_jobs: Set[str] = set()

# Base disk directory for persisting uploaded document images during asynchronous processing
BASE_STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "jobs")

# Blacklist of known synthetic placeholder tokens to reject at field normalization
FORBIDDEN_HALLUCINATIONS: List[str] = [
    "parthasarthy",
    "parthasarathy",
    "123 main street",
    "anytown",
    "123-456-7890",
    "example@domain.com",
    "9876543210",
    "09876543210",
    "dummy",
    "lorem ipsum",
    "john.doe"
]


# ==============================================================================
# Safe Deduplicated Queue Dispatcher
# ==============================================================================

async def safe_enqueue_job(job_id: str) -> bool:
    """
    Safely enqueues a job tracking ID into the execution queue while preventing
    duplicate dispatches or re-runs of already completed jobs.

    :param job_id: Unique UUID string for the job.
    :return: True if the job was newly enqueued; False if rejected as duplicate/completed.
    """
    clean_id = str(job_id).strip()
    if not clean_id:
        return False

    # 1. Guard against duplicate in-flight dispatch
    if clean_id in enqueued_or_active_jobs:
        logger.info(f"[*] Job {clean_id} is already enqueued or actively processing. Duplicate dispatch bypassed.")
        return False

    # 2. Check current status in SQLite database
    job_record = get_job(clean_id)
    if job_record:
        current_status = job_record.get("status")
        if current_status == "COMPLETED":
            logger.info(f"[*] Job {clean_id} is already COMPLETED in database. Duplicate dispatch bypassed.")
            return False
        if current_status == "PROCESSING":
            logger.info(f"[*] Job {clean_id} is currently PROCESSING. Duplicate dispatch bypassed.")
            return False

    # 3. Mark as tracked and place into queue
    enqueued_or_active_jobs.add(clean_id)
    await job_queue.put(clean_id)
    logger.info(f"[*] Job {clean_id} successfully added to background execution queue (queue size: {job_queue.qsize()}).")
    return True


# ==============================================================================
# Sanitization & Anti-Hallucination Guardrails
# ==============================================================================

def sanitize_extracted_string(val: Optional[str]) -> Optional[str]:
    """
    Sanitizes extracted text to prevent dummy or placeholder strings from persisting.
    If the value contains known placeholder tokens or empty strings, returns None.

    :param val: Raw string extracted from VLM or OCR.
    :return: Sanitized string, or None if rejected.
    """
    if not val:
        return None
    
    val_clean = str(val).strip().strip("\"'").strip()
    if not val_clean or val_clean.lower() in ["null", "none", "n/a", "unknown", "", "nil", "undefined"]:
        return None
    
    val_lower = val_clean.lower()
    for bad_token in FORBIDDEN_HALLUCINATIONS:
        if bad_token in val_lower:
            logger.warning(f"[Sanitizer] Blocked synthetic placeholder string containing '{bad_token}': {val_clean}")
            return None
            
    return val_clean


def parse_mrz_date(raw_mrz_date: Optional[str], is_expiry: bool = False) -> Optional[str]:
    """
    Converts a 6-digit ICAO 9303 MRZ date (YYMMDD) into a standard YYYY-MM-DD string.

    :param raw_mrz_date: 6-character date string in YYMMDD format.
    :param is_expiry: True if parsing an expiry date; False for birth date.
    :return: Formatted YYYY-MM-DD string, or None if parsing fails.
    """
    cleaned = sanitize_extracted_string(raw_mrz_date)
    if not cleaned or len(cleaned) != 6 or not cleaned.isdigit():
        return cleaned

    try:
        yy = int(cleaned[0:2])
        mm = int(cleaned[2:4])
        dd = int(cleaned[4:6])

        if mm < 1 or mm > 12 or dd < 1 or dd > 31:
            return cleaned

        current_year = datetime.now().year
        current_century = (current_year // 100) * 100
        current_yy = current_year % 100

        if is_expiry:
            full_year = current_century + yy if yy >= (current_yy - 10) else (current_century + 100) + yy
        else:
            full_year = current_century + yy if yy <= current_yy else (current_century - 100) + yy

        return f"{full_year:04d}-{mm:02d}-{dd:02d}"
    except Exception:
        return cleaned


# ==============================================================================
# Image Preparation Helper with Guarded Orientation Normalization
# ==============================================================================

def load_and_preprocess_image(
    file_path: str,
    max_dimension: Optional[int] = None,
    enable_auto_orient: bool = True
) -> Optional[Image.Image]:
    """
    Loads an image from disk storage, applies EXIF orientation transpositions,
    standardizes color channels to RGB, verifies orientation with confidence guardrails,
    and scales proportionally if the maximum dimension exceeds configured limits.

    :param file_path: Absolute or relative filesystem path to the image file.
    :param max_dimension: Maximum allowed width or height in pixels. Defaults to settings.max_image_dimension.
    :param enable_auto_orient: When True, uses Tesseract OSD only if confidence >= 2.0 and script is Latin.
    :return: Prepared PIL Image instance, or None if the file cannot be read.
    """
    if not os.path.exists(file_path):
        logger.warning(f"[ImageLoader] File not found on disk: {file_path}")
        return None

    target_max_dim = max_dimension if max_dimension is not None else settings.max_image_dimension

    try:
        start_t = time.perf_counter()
        raw_img = Image.open(file_path)

        # 1. Standardize orientation using EXIF orientation tags (if present)
        image = ImageOps.exif_transpose(raw_img)

        # 2. Standardize color profile to 3-channel RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 3. Guarded auto-orientation check for documents
        if enable_auto_orient:
            try:
                thumb = image.copy()
                thumb.thumbnail((800, 800), Image.Resampling.LANCZOS)
                osd_data = pytesseract.image_to_osd(thumb, output_type=pytesseract.Output.DICT)
                rotate_angle = osd_data.get("rotate", 0)
                osd_conf = float(osd_data.get("orientation_conf", 0.0))
                osd_script = osd_data.get("script", "")

                logger.info(
                    f"[ImageLoader] OSD evaluation for '{os.path.basename(file_path)}': "
                    f"rotate={rotate_angle}°, conf={osd_conf:.2f}, script={osd_script}"
                )

                if rotate_angle in [90, 180, 270] and osd_conf >= 2.0 and osd_script == "Latin":
                    logger.info(f"[ImageLoader] High-confidence OSD detected {rotate_angle}° rotation. Correcting '{os.path.basename(file_path)}'...")
                    image = image.rotate(360 - rotate_angle, expand=True)
                elif rotate_angle != 0:
                    logger.info(
                        f"[ImageLoader] Bypassing OSD {rotate_angle}° rotation due to low confidence ({osd_conf:.2f} < 2.0) "
                        f"or non-Latin script ('{osd_script}'). Keeping native orientation."
                    )
            except Exception as osd_err:
                logger.debug(f"[ImageLoader] OSD orientation detection bypassed: {osd_err}")

        orig_w, orig_h = image.size

        # 4. Downscale proportionally if larger than configured ceiling
        if max(orig_w, orig_h) > target_max_dim:
            image.thumbnail((target_max_dim, target_max_dim), Image.Resampling.LANCZOS)
            new_w, new_h = image.size
            logger.info(
                f"[ImageLoader] Rescaled '{os.path.basename(file_path)}' from {orig_w}x{orig_h} "
                f"to {new_w}x{new_h} in {(time.perf_counter() - start_t):.3f}s."
            )
        else:
            logger.info(
                f"[ImageLoader] Loaded '{os.path.basename(file_path)}' at native {orig_w}x{orig_h} "
                f"in {(time.perf_counter() - start_t):.3f}s."
            )

        return image
    except Exception as exc:
        logger.error(f"[ImageLoader] Failed to load/preprocess '{file_path}': {str(exc)}", exc_info=True)
        return None


# ==============================================================================
# Inference Subroutine Helpers
# ==============================================================================

def run_vlm_inference(
    image: Image.Image,
    system_prompt: str,
    ml_state: dict,
    stage_label: str = "VLM",
    max_tokens_override: Optional[int] = None,
    min_pixels_override: Optional[int] = None,
    max_pixels_override: Optional[int] = None
) -> dict:
    """
    Executes a structured query against the shared in-memory Qwen2-VL model instance
    with deterministic greedy decoding, custom visual patch allocations, and strict JSON parsing.

    :param image: Preprocessed PIL Image instance or cropped section.
    :param system_prompt: Guidance text instructing the VLM on expected JSON fields.
    :param ml_state: Shared dictionary holding 'model' and 'processor' instances.
    :param stage_label: Diagnostic label for logging (e.g. 'Passport Bio', 'Visa', etc.).
    :param max_tokens_override: Optional integer to override default max_new_tokens.
    :param min_pixels_override: Optional visual patch floor in pixels.
    :param max_pixels_override: Optional visual patch ceiling in pixels.
    :return: Extracted key-value dictionary parsed from the model's generated JSON.
    """
    from qwen_vl_utils import process_vision_info

    processor = ml_state.get("processor")
    model = ml_state.get("model")

    if not model or not processor:
        raise RuntimeError("Qwen2-VL model and processor are not loaded in ml_state.")

    # Explicitly pin PyTorch compute threads to physical CPU cores
    torch.set_num_threads(settings.torch_cpu_threads)

    effective_min_pixels = min_pixels_override if min_pixels_override is not None else settings.min_pixels
    effective_max_pixels = max_pixels_override if max_pixels_override is not None else settings.max_pixels

    start_vlm_time = time.perf_counter()
    logger.info(
        f"[{stage_label}] Preparing prompt tensor inputs (image: {image.size[0]}x{image.size[1]}, "
        f"patches: {effective_min_pixels}-{effective_max_pixels}px)..."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": effective_min_pixels,
                    "max_pixels": effective_max_pixels
                },
                {"type": "text", "text": system_prompt}
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )

    inputs = {k: v.to(settings.device) for k, v in inputs.items()}

    effective_max_tokens = max_tokens_override if max_tokens_override is not None else settings.max_new_tokens
    logger.info(f"[{stage_label}] Beginning PyTorch CPU token generation (max_new_tokens={effective_max_tokens})...")
    gen_start = time.perf_counter()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=effective_max_tokens,
            do_sample=False
        )

    gen_duration = time.perf_counter() - gen_start
    total_duration = time.perf_counter() - start_vlm_time

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    tokens_generated = len(generated_ids_trimmed[0]) if generated_ids_trimmed else 0

    raw_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    logger.info(
        f"[{stage_label}] Generated {tokens_generated} tokens in {gen_duration:.2f}s "
        f"({tokens_generated / max(0.01, gen_duration):.1f} tok/s). Total stage latency: {total_duration:.2f}s."
    )

    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            parsed_json = json.loads(match.group(0))
            return parsed_json
        return json.loads(raw_text)
    except Exception as parse_err:
        logger.warning(
            f"[{stage_label}] JSON parse error: {parse_err}. Attempting truncation cleanup. Excerpt: {raw_text[:200]}"
        )
        try:
            cleaned = re.sub(r",\s*([\}\]])", r"\1", raw_text)
            if not cleaned.strip().endswith("}"):
                cleaned = cleaned.strip() + '"}}'
            return json.loads(cleaned)
        except Exception:
            return {"confidence_note": f"Raw model response: {raw_text}"}


# ==============================================================================
# Normalization & Tournament Helpers
# ==============================================================================

def normalize_vlm_field(
    raw_field: Any,
    default_score: float = 0.85,
    source: str = "qwen2_vl"
) -> FieldResult:
    """
    Parses a field returned by Qwen2-VL into a validated FieldResult.
    Sanitizes against dummy strings and synthetic placeholder tokens.

    :param raw_field: Dict containing {'value': ..., 'confidence': ...} or a raw scalar.
    :param default_score: Default confidence score if none is specified by the model.
    :param source: Pipeline provenance label.
    :return: Sanitized FieldResult instance.
    """
    if raw_field is None:
        return FieldResult(value=None, confidence=0.0, source=source)

    if isinstance(raw_field, dict):
        val = raw_field.get("value")
        raw_conf = raw_field.get("confidence", default_score)
    else:
        val = str(raw_field)
        raw_conf = default_score

    clean_val = sanitize_extracted_string(val)
    if clean_val is None:
        return FieldResult(value=None, confidence=0.0, source=source)

    try:
        conf_float = float(raw_conf)
        conf_float = max(0.0, min(1.0, conf_float))
    except (ValueError, TypeError):
        conf_float = default_score

    return FieldResult(
        value=clean_val,
        confidence=round(conf_float, 2),
        source=source
    )


def select_best_field(
    field_name: str,
    cand_mrz: FieldResult,
    cand_vlm: FieldResult,
    notes: list
) -> FieldResult:
    """
    Conducts a confidence tournament for a bio field between Tesseract MRZ and Qwen2-VL.

    :param field_name: Name of the target field for diagnostic logging.
    :param cand_mrz: Extraction result produced by Tesseract MRZ.
    :param cand_vlm: Extraction result produced by Qwen2-VL.
    :param notes: In-out list collecting execution notes.
    :return: Winning FieldResult.
    """
    if cand_mrz.value and not cand_vlm.value:
        return cand_mrz
    if cand_vlm.value and not cand_mrz.value:
        return cand_vlm
    if not cand_mrz.value and not cand_vlm.value:
        return FieldResult(value=None, confidence=0.0, source="unassigned")

    if cand_mrz.confidence >= cand_vlm.confidence:
        winning = cand_mrz
        notes.append(
            f"Field '{field_name}': MRZ won ({cand_mrz.confidence:.2f} vs VLM {cand_vlm.confidence:.2f})."
        )
    else:
        winning = cand_vlm
        notes.append(
            f"Field '{field_name}': VLM won ({cand_vlm.confidence:.2f} vs MRZ {cand_mrz.confidence:.2f})."
        )

    return winning


def refresh_document_paths(job_id: str) -> Dict[str, str]:
    """
    Queries SQLite and inspects the on-disk job directory to discover attached files.
    Supports passport_bio, visa_document, visa_stamp, legacy visa_page, entry_stamp, and welcome_form.

    :param job_id: UUID tracking string.
    :return: Mapping of document types to verified local file paths.
    """
    job_record = get_job(job_id)
    doc_paths: Dict[str, str] = job_record.get("doc_paths", {}) if job_record else {}

    job_dir = os.path.join(BASE_STORAGE_DIR, job_id)
    if os.path.isdir(job_dir):
        for filename in os.listdir(job_dir):
            full_path = os.path.join(job_dir, filename)
            if not os.path.isfile(full_path):
                continue
            if (filename.startswith("bio_") or filename.startswith("passport_bio_")) and "passport_bio" not in doc_paths:
                doc_paths["passport_bio"] = full_path
            elif filename.startswith("visadoc_") and "visa_document" not in doc_paths:
                doc_paths["visa_document"] = full_path
            elif filename.startswith("visastamp_") and "visa_stamp" not in doc_paths:
                doc_paths["visa_stamp"] = full_path
            elif (filename.startswith("visa_") or filename.startswith("visa_page_")) and "visa_page" not in doc_paths:
                doc_paths["visa_page"] = full_path
            elif (filename.startswith("stamp_") or filename.startswith("entry_stamp_")) and "entry_stamp" not in doc_paths:
                doc_paths["entry_stamp"] = full_path
            elif (filename.startswith("welcome_") or filename.startswith("welcome_form_") or filename.startswith("form_c_")) and "welcome_form" not in doc_paths:
                doc_paths["welcome_form"] = full_path

    return doc_paths


# ==============================================================================
# Job Processing Pipeline
# ==============================================================================

async def process_single_job(job_input: Union[str, Dict[str, Any]], ml_state: dict) -> None:
    """
    Executes the multi-document extraction pipeline for a single queued job.
    Uses neutral administrative framing, high-definition visual patches, and
    OCR-assisted hints on Form-C to decipher cursive handwriting.
    Populates both the fused composite record and isolated per-document data buckets.

    :param job_input: Either a UUID string or a recovered dictionary holding 'job_id'.
    :param ml_state: Shared dictionary holding model handles.
    """
    if isinstance(job_input, dict):
        job_id = str(job_input.get("job_id", "")).strip()
    else:
        job_id = str(job_input).strip()

    if not job_id:
        logger.error("[!] Dispatched empty or invalid job identifier to process_single_job.")
        return

    # Guard against duplicate execution: if already completed, bypass immediately
    job_record = get_job(job_id)
    if job_record and job_record.get("status") == "COMPLETED":
        logger.info(f"[*] Job {job_id} is already COMPLETED. Skipping duplicate queue execution.")
        enqueued_or_active_jobs.discard(job_id)
        return

    job_start_time = time.perf_counter()
    logger.info(f"[*] Starting extraction pipeline for job: {job_id}")
    update_job_status(job_id, "PROCESSING")

    fused_record = GuestFormCRecord()
    fused_record.documents = DocumentExtractionContainer()
    processing_notes: List[str] = []

    try:
        # ======================================================================
        # 1. PASSPORT BIO PAGE: 4-ANGLE MRZ TOURNAMENT + UPRIGHT VLM [TASK 50]
        # ======================================================================
        doc_paths = refresh_document_paths(job_id)
        bio_path = doc_paths.get("passport_bio")

        if bio_path and os.path.exists(bio_path):
            logger.info(f"[{job_id}] Stage 1/4: Dual-engine extraction for passport bio page ({bio_path})...")

            # Load base image with EXIF orientation baked in
            base_bio_img = Image.open(bio_path)
            base_bio_img = ImageOps.exif_transpose(base_bio_img)
            if base_bio_img.mode != "RGB":
                base_bio_img = base_bio_img.convert("RGB")

            # --- Engine A: 4-Angle MRZ Scan Tournament (Task 50) ---
            mrz_start = time.perf_counter()
            best_mrz_record = None
            best_score = -1
            best_rotation_angle = 0
            valid_mrz_flag = False

            # Test 0, 90, 180, 270 degrees rotations
            for test_angle in [0, 90, 180, 270]:
                rot_img = base_bio_img.rotate(test_angle, expand=True) if test_angle != 0 else base_bio_img
                buf = io.BytesIO()
                rot_img.save(buf, format="JPEG", quality=90)
                buf.seek(0)

                try:
                    cand_mrz = await asyncio.to_thread(read_mrz, buf)
                    if cand_mrz is not None:
                        cand_score = int(getattr(cand_mrz, "valid_score", 0))
                        cand_valid_composite = bool(getattr(cand_mrz, "valid_composite", False))
                        cand_valid_number = bool(getattr(cand_mrz, "valid_number", False))
                        cand_is_valid = bool(cand_valid_composite or (cand_score >= 80 and cand_valid_number))

                        logger.info(f"[{job_id}] MRZ scan at {test_angle}°: score={cand_score}, valid={cand_is_valid}")

                        if cand_score > best_score:
                            best_score = cand_score
                            best_mrz_record = cand_mrz
                            best_rotation_angle = test_angle
                            valid_mrz_flag = cand_is_valid

                        # If 100% valid MRZ checksum obtained, stop searching other angles
                        if cand_is_valid and cand_score >= 85:
                            break
                except Exception as angle_mrz_err:
                    logger.debug(f"[{job_id}] MRZ angle {test_angle}° error: {angle_mrz_err}")

            mrz_candidates: Dict[str, FieldResult] = {}
            if best_mrz_record is not None:
                mrz_dict = best_mrz_record.to_dict() if hasattr(best_mrz_record, "to_dict") else {}
                base_text_conf = round(max(0.0, min(1.0, best_score / 100.0)), 2)

                check_number = bool(getattr(best_mrz_record, "valid_number", False))
                check_dob = bool(getattr(best_mrz_record, "valid_date_of_birth", False))
                check_exp = bool(getattr(best_mrz_record, "valid_expiration_date", False))

                def mk_checked(val: Optional[str], check_passed: bool, is_date: bool = False, is_expiry: bool = False) -> FieldResult:
                    clean = sanitize_extracted_string(val)
                    if not clean:
                        return FieldResult(value=None, confidence=0.0, source="tesseract_mrz")
                    if is_date:
                        clean = parse_mrz_date(clean, is_expiry=is_expiry) or clean

                    return FieldResult(
                        value=clean,
                        confidence=0.95 if check_passed else 0.30,
                        source="tesseract_mrz"
                    )

                def mk_text(val: Optional[str]) -> FieldResult:
                    clean = sanitize_extracted_string(val)
                    if not clean:
                        return FieldResult(value=None, confidence=0.0, source="tesseract_mrz")
                    return FieldResult(
                        value=clean,
                        confidence=base_text_conf,
                        source="tesseract_mrz"
                    )

                mrz_candidates["surname"] = mk_text(mrz_dict.get("surname"))
                mrz_candidates["given_names"] = mk_text(mrz_dict.get("names"))
                mrz_candidates["passport_number"] = mk_checked(mrz_dict.get("number"), check_number)
                mrz_candidates["nationality"] = mk_text(mrz_dict.get("nationality"))
                mrz_candidates["date_of_birth"] = mk_checked(mrz_dict.get("date_of_birth"), check_dob, is_date=True, is_expiry=False)
                mrz_candidates["sex"] = mk_text(mrz_dict.get("sex"))
                mrz_candidates["passport_expiry_date"] = mk_checked(mrz_dict.get("expiration_date"), check_exp, is_date=True, is_expiry=True)

                mrz_duration = time.perf_counter() - mrz_start
                logger.info(
                    f"[{job_id}] Tesseract MRZ completed in {mrz_duration:.2f}s "
                    f"(winning angle={best_rotation_angle}°, valid={valid_mrz_flag}, score={best_score})."
                )
                processing_notes.append(f"MRZ parsed at {best_rotation_angle}° in {mrz_duration:.2f}s (valid: {valid_mrz_flag}, score: {best_score}).")
            else:
                mrz_duration = time.perf_counter() - mrz_start
                logger.info(f"[{job_id}] Tesseract MRZ found no patterns across 4 angles ({mrz_duration:.2f}s).")
                processing_notes.append(f"MRZ not detected across 4 rotations ({mrz_duration:.2f}s).")

            # --- Engine B: Qwen2-VL Visual Bio on Upright Image (Task 50) ---
            # Rotate bio image to match winning MRZ angle so Qwen2-VL receives upright image
            if best_rotation_angle != 0:
                logger.info(f"[{job_id}] Aligning bio image to winning angle {best_rotation_angle}° for VLM inference.")
                bio_image = base_bio_img.rotate(best_rotation_angle, expand=True)
            else:
                bio_image = base_bio_img

            # Scale to max dimension if needed
            if max(bio_image.size) > settings.max_image_dimension:
                bio_image.thumbnail((settings.max_image_dimension, settings.max_image_dimension), Image.Resampling.LANCZOS)

            vlm_candidates: Dict[str, FieldResult] = {}
            bio_prompt = (
                "You are an administrative document verification assistant reading a passport bio page image.\n"
                "Extract the printed text into a valid JSON object matching this schema:\n"
                "{\n"
                '  "surname": {"value": string or null, "confidence": float},\n'
                '  "given_names": {"value": string or null, "confidence": float},\n'
                '  "passport_number": {"value": string or null, "confidence": float},\n'
                '  "nationality": {"value": string or null, "confidence": float},\n'
                '  "date_of_birth": {"value": string or null, "confidence": float},\n'
                '  "sex": {"value": "M" or "F" or null, "confidence": float},\n'
                '  "expiry_date": {"value": string or null, "confidence": float}\n'
                "}\n"
                "Instructions:\n"
                "- Transcribe only text clearly legible on this document.\n"
                "- For passport_number, transcribe the exact alphanumeric string printed in the Visual Inspection Zone (e.g. C9K82159L).\n"
                "- If any field is missing or unreadable, return null for value and 0.0 for confidence.\n"
                "- Output JSON only."
            )
            vlm_bio = await asyncio.to_thread(run_vlm_inference, bio_image, bio_prompt, ml_state, "Passport Bio VLM")

            vlm_candidates["surname"] = normalize_vlm_field(vlm_bio.get("surname"))
            vlm_candidates["given_names"] = normalize_vlm_field(vlm_bio.get("given_names"))
            vlm_candidates["passport_number"] = normalize_vlm_field(vlm_bio.get("passport_number"))
            vlm_candidates["nationality"] = normalize_vlm_field(vlm_bio.get("nationality"))
            vlm_candidates["date_of_birth"] = normalize_vlm_field(vlm_bio.get("date_of_birth"))
            vlm_candidates["sex"] = normalize_vlm_field(vlm_bio.get("sex"))
            vlm_candidates["passport_expiry_date"] = normalize_vlm_field(vlm_bio.get("expiry_date"))

            processing_notes.append("Qwen2-VL visual bio extraction completed.")

            # --- Tournament Selection ---
            empty_field = FieldResult(value=None, confidence=0.0, source="none")
            fused_record.valid_mrz = valid_mrz_flag

            fused_record.surname = select_best_field(
                "surname",
                mrz_candidates.get("surname", empty_field),
                vlm_candidates.get("surname", empty_field),
                processing_notes
            )
            fused_record.given_names = select_best_field(
                "given_names",
                mrz_candidates.get("given_names", empty_field),
                vlm_candidates.get("given_names", empty_field),
                processing_notes
            )
            fused_record.passport_number = select_best_field(
                "passport_number",
                mrz_candidates.get("passport_number", empty_field),
                vlm_candidates.get("passport_number", empty_field),
                processing_notes
            )
            fused_record.nationality = select_best_field(
                "nationality",
                mrz_candidates.get("nationality", empty_field),
                vlm_candidates.get("nationality", empty_field),
                processing_notes
            )
            fused_record.date_of_birth = select_best_field(
                "date_of_birth",
                mrz_candidates.get("date_of_birth", empty_field),
                vlm_candidates.get("date_of_birth", empty_field),
                processing_notes
            )
            fused_record.sex = select_best_field(
                "sex",
                mrz_candidates.get("sex", empty_field),
                vlm_candidates.get("sex", empty_field),
                processing_notes
            )
            fused_record.passport_expiry_date = select_best_field(
                "passport_expiry_date",
                mrz_candidates.get("passport_expiry_date", empty_field),
                vlm_candidates.get("passport_expiry_date", empty_field),
                processing_notes
            )

            # Task 45: Persist isolated PassportBioRecord
            fused_record.documents.passport_bio = PassportBioRecord(
                surname=fused_record.surname,
                given_names=fused_record.given_names,
                sex=fused_record.sex,
                date_of_birth=fused_record.date_of_birth,
                nationality=fused_record.nationality,
                passport_number=fused_record.passport_number,
                passport_expiry_date=fused_record.passport_expiry_date,
                valid_mrz=fused_record.valid_mrz
            )

            fused_record.processing_notes = list(processing_notes)
            save_intermediate_job_result(job_id, fused_record.model_dump())
            logger.info(f"[{job_id}] Stage 1/4 completed and intermediate state persisted to database.")
        else:
            logger.warning(f"[{job_id}] Passport bio image not found on disk: {bio_path}")
            processing_notes.append("Passport bio image missing from job directory.")

        # ======================================================================
        # 2A. VISA DOCUMENT (E-VISA PAPER / PDF PRINT) - TASK 42
        # ======================================================================
        doc_paths = refresh_document_paths(job_id)
        visa_doc_path = doc_paths.get("visa_document")

        if visa_doc_path and os.path.exists(visa_doc_path):
            logger.info(f"[{job_id}] Stage 2A: Running Qwen2-VL e-Visa document extraction ({visa_doc_path})...")
            vdoc_image = await asyncio.to_thread(load_and_preprocess_image, visa_doc_path)
            if vdoc_image:
                vdoc_prompt = (
                    "You are an administrative document verification assistant inspecting an Indian e-Visa paper printout or OCI card document image.\n"
                    "Extract only the text physically printed on the document into a valid JSON object matching this schema:\n"
                    "{\n"
                    '  "visa_number": {"value": string or null, "confidence": float},\n'
                    '  "visa_type": {"value": string or null, "confidence": float},\n'
                    '  "issue_date": {"value": string or null, "confidence": float},\n'
                    '  "expiry_date": {"value": string or null, "confidence": float},\n'
                    '  "place_of_issue": {"value": string or null, "confidence": float}\n'
                    "}\n"
                    "Instructions:\n"
                    "- For visa_number, look for 'Visa No', 'Document No', or the bold alphanumeric identifier.\n"
                    "- For visa_type, look for the category such as 'TOURIST', 'T', 'BUSINESS', 'X', 'OCI', or 'STUDENT'.\n"
                    "- If an expiry date is not printed on the document, return null for expiry_date.\n"
                    "- For place_of_issue, transcribe the exact issuing mission or authority printed on the document.\n"
                    "- If any field is missing or unreadable, return null for value and 0.0 for confidence.\n"
                    "- Output JSON only."
                )
                vlm_vdoc = await asyncio.to_thread(run_vlm_inference, vdoc_image, vdoc_prompt, ml_state, "Visa Document VLM")
                
                fused_record.documents.visa_document = VisaDocumentRecord(
                    visa_number=normalize_vlm_field(vlm_vdoc.get("visa_number"), source="visa_document"),
                    visa_type=normalize_vlm_field(vlm_vdoc.get("visa_type"), source="visa_document"),
                    visa_issue_date=normalize_vlm_field(vlm_vdoc.get("issue_date"), source="visa_document"),
                    visa_expiry_date=normalize_vlm_field(vlm_vdoc.get("expiry_date"), source="visa_document"),
                    visa_place_of_issue=normalize_vlm_field(vlm_vdoc.get("place_of_issue"), source="visa_document")
                )
                
                fused_record.visa_number = fused_record.documents.visa_document.visa_number
                fused_record.visa_type = fused_record.documents.visa_document.visa_type
                fused_record.visa_issue_date = fused_record.documents.visa_document.visa_issue_date
                fused_record.visa_expiry_date = fused_record.documents.visa_document.visa_expiry_date
                fused_record.visa_place_of_issue = fused_record.documents.visa_document.visa_place_of_issue
                
                processing_notes.append("e-Visa document details extracted via Qwen2-VL.")
                fused_record.processing_notes = list(processing_notes)
                save_intermediate_job_result(job_id, fused_record.model_dump())
                logger.info(f"[{job_id}] Stage 2A completed and intermediate state persisted to database.")

        # ======================================================================
        # 2B. VISA STAMP (PASSPORT BOOKLET CONSULAR STICKER/STAMP) - TASK 42
        # ======================================================================
        doc_paths = refresh_document_paths(job_id)
        visa_stamp_path = doc_paths.get("visa_stamp") or doc_paths.get("visa_page")

        if visa_stamp_path and os.path.exists(visa_stamp_path):
            logger.info(f"[{job_id}] Stage 2B: Running Qwen2-VL passport booklet visa stamp extraction ({visa_stamp_path})...")
            vstamp_image = await asyncio.to_thread(load_and_preprocess_image, visa_stamp_path)
            if vstamp_image:
                vstamp_prompt = (
                    "You are an administrative document verification assistant inspecting a physical consular visa sticker or stamp on a passport page.\n"
                    "Extract only the text physically printed on the visa sticker into a valid JSON object matching this schema:\n"
                    "{\n"
                    '  "visa_number": {"value": string or null, "confidence": float},\n'
                    '  "visa_type": {"value": string or null, "confidence": float},\n'
                    '  "issue_date": {"value": string or null, "confidence": float},\n'
                    '  "expiry_date": {"value": string or null, "confidence": float},\n'
                    '  "place_of_issue": {"value": string or null, "confidence": float}\n'
                    "}\n"
                    "Instructions:\n"
                    "- For visa_number, look for 'Visa No', 'Document No', or the bold 7 to 9 character alphanumeric identifier.\n"
                    "- For visa_type, look for 'TOURIST', 'T', 'BUSINESS', 'X', 'OCI', or 'STUDENT'.\n"
                    "- If an expiry date is not printed, return null for expiry_date.\n"
                    "- For place_of_issue, transcribe the exact issuing mission or authority printed on the sticker.\n"
                    "- If any field is missing or unreadable, return null for value and 0.0 for confidence.\n"
                    "- Output JSON only."
                )
                vlm_vstamp = await asyncio.to_thread(run_vlm_inference, vstamp_image, vstamp_prompt, ml_state, "Visa Stamp VLM")
                
                fused_record.documents.visa_stamp = VisaStampRecord(
                    visa_number=normalize_vlm_field(vlm_vstamp.get("visa_number"), source="visa_stamp"),
                    visa_type=normalize_vlm_field(vlm_vstamp.get("visa_type"), source="visa_stamp"),
                    visa_issue_date=normalize_vlm_field(vlm_vstamp.get("issue_date"), source="visa_stamp"),
                    visa_expiry_date=normalize_vlm_field(vlm_vstamp.get("expiry_date"), source="visa_stamp"),
                    visa_place_of_issue=normalize_vlm_field(vlm_vstamp.get("place_of_issue"), source="visa_stamp")
                )
                
                if not fused_record.visa_number.value:
                    fused_record.visa_number = fused_record.documents.visa_stamp.visa_number
                    fused_record.visa_type = fused_record.documents.visa_stamp.visa_type
                    fused_record.visa_issue_date = fused_record.documents.visa_stamp.visa_issue_date
                    fused_record.visa_expiry_date = fused_record.documents.visa_stamp.visa_expiry_date
                    fused_record.visa_place_of_issue = fused_record.documents.visa_stamp.visa_place_of_issue

                processing_notes.append("Passport booklet visa stamp details extracted via Qwen2-VL.")
                fused_record.processing_notes = list(processing_notes)
                save_intermediate_job_result(job_id, fused_record.model_dump())
                logger.info(f"[{job_id}] Stage 2B completed and intermediate state persisted to database.")

        # ======================================================================
        # 3. IMMIGRATION ENTRY STAMP EXTRACTION (TASK 49: MULTI-ANGLE GUARD)
        # ======================================================================
        doc_paths = refresh_document_paths(job_id)
        stamp_path = doc_paths.get("entry_stamp")

        if stamp_path and os.path.exists(stamp_path):
            logger.info(f"[{job_id}] Stage 3/4: Initializing Entry Stamp extraction ({stamp_path})...")
            stamp_image = await asyncio.to_thread(load_and_preprocess_image, stamp_path)
            
            if stamp_image:
                best_stamp_angle = 0
                stamp_keywords = [
                    "IMMIGRATION", "ARRIVAL", "ARRIVED", "ENTRY", "INDIA",
                    "COCHIN", "MUMBAI", "DELHI", "BANGALORE", "CHENNAI",
                    "JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
                ]

                try:
                    thumb_stamp = stamp_image.copy()
                    thumb_stamp.thumbnail((800, 800), Image.Resampling.LANCZOS)
                    enhancer = ImageEnhance.Contrast(thumb_stamp)
                    thumb_enhanced = enhancer.enhance(1.8)
                    
                    found_angle = None
                    max_matches = 0
                    
                    for test_angle in [0, 90, 180, 270]:
                        rotated_thumb = thumb_enhanced.rotate(test_angle, expand=True) if test_angle != 0 else thumb_enhanced
                        ocr_txt = await asyncio.to_thread(pytesseract.image_to_string, rotated_thumb)
                        ocr_upper = ocr_txt.upper()
                        
                        match_count = sum(1 for kw in stamp_keywords if kw in ocr_upper)
                        if match_count > max_matches:
                            max_matches = match_count
                            found_angle = test_angle

                    if found_angle is not None and max_matches >= 1:
                        best_stamp_angle = found_angle
                        logger.info(f"[{job_id}] Rectangular entry stamp ink matched {max_matches} keywords at {best_stamp_angle}° rotation.")
                        if best_stamp_angle != 0:
                            stamp_image = stamp_image.rotate(best_stamp_angle, expand=True)
                    else:
                        logger.info(f"[{job_id}] Stamp orientation check retained native 0°.")
                except Exception as stamp_orient_err:
                    logger.warning(f"[{job_id}] Stamp orientation detection bypassed: {stamp_orient_err}")

                stamp_prompt = (
                    "You are an administrative document verification assistant inspecting a passport page for an Indian immigration entry stamp.\n"
                    "INSTRUCTIONS ON STAMP STRUCTURE:\n"
                    "- The stamp is typically a RECTANGULAR, OVAL, or SQUARE colored ink impression (blue, green, purple, or black ink).\n"
                    "- Look for the rectangular border containing text such as 'INDIAN IMMIGRATION', 'BUREAU OF IMMIGRATION', or 'ARRIVAL'.\n"
                    "- Top / Outer border: Contains 'INDIAN IMMIGRATION' or 'INDIA'.\n"
                    "- Center of stamp: Contains the stamped date (e.g. '14 OCT 2024', '14-10-2024', '07/07/20', or '26 JUL 2020').\n"
                    "- Bottom border: Contains the port or airport name, e.g. 'COCHIN AIRPORT', 'MUMBAI AIRPORT', 'DELHI', 'CHENNAI AIRPORT'.\n\n"
                    "CRITICAL: Ignore other visas or printed passport booklet text. Focus strictly on the stamped colored ink arrival box.\n\n"
                    "Extract the arrival details into a valid JSON object matching this schema:\n"
                    "{\n"
                    '  "arrival_date": {"value": "YYYY-MM-DD" or string or null, "confidence": float},\n'
                    '  "arrival_port": {"value": string or null, "confidence": float}\n'
                    "}\n"
                    "Notes:\n"
                    "- If date is readable in format DD-MM-YYYY, DD/MM/YY, or DD MMM YYYY, normalize to standard date string.\n"
                    "- If port of entry name is visible on the stamp, extract it with high confidence (e.g. 'COCHIN AIRPORT').\n"
                    "- If unreadable or missing, return null for value and 0.0 for confidence.\n"
                    "- Output JSON only."
                )

                vlm_stamp = await asyncio.to_thread(
                    run_vlm_inference,
                    stamp_image,
                    stamp_prompt,
                    ml_state,
                    "Stamp VLM (Rectangular Guard)",
                    256,
                    38416,
                    150528
                )
                
                fused_record.documents.entry_stamp = EntryStampRecord(
                    arrival_date_india=normalize_vlm_field(vlm_stamp.get("arrival_date"), source="entry_stamp"),
                    arrival_port_india=normalize_vlm_field(vlm_stamp.get("arrival_port"), source="entry_stamp")
                )
                
                fused_record.arrival_date_india = fused_record.documents.entry_stamp.arrival_date_india
                fused_record.arrival_port_india = fused_record.documents.entry_stamp.arrival_port_india
                
                processing_notes.append("Entry stamp details extracted via Qwen2-VL.")
                fused_record.processing_notes = list(processing_notes)
                save_intermediate_job_result(job_id, fused_record.model_dump())
                logger.info(f"[{job_id}] Stage 3/4 completed and intermediate state persisted to database.")
        else:
            logger.info(f"[{job_id}] Stage 3/4: No entry_stamp document attached to job. Skipping.")

        # ======================================================================
        # 4. HANDWRITTEN FORM-C ARRIVAL REPORT (OCR-GROUNDED UNIFIED PASS)
        # ======================================================================
        doc_paths = refresh_document_paths(job_id)
        welcome_path = doc_paths.get("welcome_form")

        if welcome_path and os.path.exists(welcome_path):
            logger.info(f"[{job_id}] Stage 4/4: Initializing OCR-grounded full-page Form-C extraction ({welcome_path})...")
            welcome_raw = await asyncio.to_thread(load_and_preprocess_image, welcome_path)
            
            if welcome_raw:
                try:
                    sample_thumb = welcome_raw.copy()
                    sample_thumb.thumbnail((800, 800))
                    t_w, t_h = sample_thumb.size

                    test_top_crop = sample_thumb.crop((0, 0, t_w, int(t_h * 0.35)))
                    txt_top_native = await asyncio.to_thread(pytesseract.image_to_string, test_top_crop)
                    has_header_native = "ARRIVAL" in txt_top_native.upper() or "FORM C" in txt_top_native.upper()

                    if not has_header_native:
                        test_flipped = sample_thumb.rotate(180)
                        test_flipped_crop = test_flipped.crop((0, 0, t_w, int(t_h * 0.35)))
                        txt_top_flipped = await asyncio.to_thread(pytesseract.image_to_string, test_flipped_crop)
                        has_header_flipped = "ARRIVAL" in txt_top_flipped.upper() or "FORM C" in txt_top_flipped.upper()

                        if has_header_flipped:
                            logger.info(f"[{job_id}] Form-C header detected upside down. Rotating 180° to upright orientation.")
                            welcome_raw = welcome_raw.rotate(180, expand=True)
                        else:
                            logger.info(f"[{job_id}] Form-C header check inconclusive. Preserving native orientation.")
                    else:
                        logger.info(f"[{job_id}] Form-C header verified upright in native orientation.")
                except Exception as orient_err:
                    logger.warning(f"[{job_id}] Form-C header orientation test bypassed: {orient_err}")

                img_w, img_h = welcome_raw.size
                logger.info(f"[{job_id}] Running unified full-document extraction on Form-C ({img_w}x{img_h})...")

                ocr_full_text = ""
                try:
                    ocr_full_raw = await asyncio.to_thread(pytesseract.image_to_string, welcome_raw)
                    ocr_full_lines = [line.strip() for line in ocr_full_raw.splitlines() if line.strip()]
                    ocr_full_text = "\n".join(ocr_full_lines)
                except Exception:
                    ocr_full_text = "No OCR hints available."

                welcome_full_prompt = (
                    "You are an administrative office assistant digitizing an official Form C (Rule 14) Arrival Report homestay registration card.\n"
                    "CRITICAL ANTI-HALLUCINATION INSTRUCTION: The card contains pre-printed homestay letterhead at the top ('HOMESTED, 1386 A, PARVATHY NILAYAM, THAMARAKULAM ROAD, KOCHI - 682 001'). "
                    "Under NO circumstances extract 'HOMESTED', 'PARVATHY NILAYAM', 'KOCHI', or 'India' as the guest's address or city. IGNORE that homestay header completely.\n"
                    "Extract ONLY the foreigner guest's genuine handwritten entries in Rows 1 to 12 into this JSON schema:\n"
                    "{\n"
                    '  "name_of_foreigner": {"value": string or null, "confidence": float},\n'
                    '  "full_address": {"value": string or null, "confidence": float},\n'
                    '  "city": {"value": string or null, "confidence": float},\n'
                    '  "country": {"value": string or null, "confidence": float},\n'
                    '  "email": {"value": string or null, "confidence": float},\n'
                    '  "telephone_mobile": {"value": string or null, "confidence": float},\n'
                    '  "nationality": {"value": string or null, "confidence": float},\n'
                    '  "place_of_issue_of_visa": {"value": string or null, "confidence": float},\n'
                    '  "date_of_arrival_in_india": {"value": "YYYY-MM-DD" or null, "confidence": float},\n'
                    '  "datetime_arrival_homestay": {"value": string or null, "confidence": float},\n'
                    '  "arrived_from": {"value": string or null, "confidence": float},\n'
                    '  "proceeding_to": {"value": string or null, "confidence": float},\n'
                    '  "proposed_duration_stay_india": {"value": string or null, "confidence": float},\n'
                    '  "purpose_of_visit_profession": {"value": string or null, "confidence": float},\n'
                    '  "expected_date_departure": {"value": "YYYY-MM-DD" or null, "confidence": float},\n'
                    '  "country_city_arrived_to_india": {"value": string or null, "confidence": float}\n'
                    "}\n\n"
                    "Numbered Row Breakdown on the Form:\n"
                    "- Row 1: Name of foreigner (Surname First)\n"
                    "- Row 2: Contains 3 sub-lines for the foreigner's permanent foreign residence:\n"
                    "    * Line 1: 'Full Address' (handwritten street address in home country)\n"
                    "    * Line 2: 'City' (handwritten foreign town) and 'Country' (handwritten foreign home country)\n"
                    "    * Line 3: 'E-mail' and 'Telphone / Mobile'\n"
                    "- Row 3: Nationality\n"
                    "- Row 4: Place of issue of visa\n"
                    "- Row 5: Date of Arrival in India\n"
                    "- Row 6: Date and Time of Arrival in the home stay (e.g. '09.10.16 18:00' or '17/5/2026 12:00')\n"
                    "- Row 7: Arrived from (city, state, or country written in row 7)\n"
                    "- Row 8: Proceeding to (from here) (city, state, or country written in row 8)\n"
                    "- Row 9: Proposed duration of stay in India (e.g. '2 days', '10 days', '1 month')\n"
                    "- Row 10: Purpose of visit & your Profession (e.g. 'Tourist', 'Business')\n"
                    "- Row 11: Expected date of departure (from here) (date written in row 11)\n"
                    "- Row 12: Country & city you arrived to India (written in row 12)\n\n"
                    "Here are raw OCR character tokens detected on the upright form to assist your reading:\n"
                    f"\"\"\"\n{ocr_full_text}\n\"\"\"\n\n"
                    "Instructions:\n"
                    "- Transcribe only genuine handwriting. If a row or sub-line is blank, return null for value and 0.0 for confidence.\n"
                    "- Output JSON only."
                )

                vlm_form = await asyncio.to_thread(
                    run_vlm_inference,
                    welcome_raw,
                    welcome_full_prompt,
                    ml_state,
                    "Form-C Full Document VLM",
                    448,
                    65536,
                    262144
                )

                fused_record.documents.welcome_form = FormCReportRecord(
                    name_of_foreigner=normalize_vlm_field(vlm_form.get("name_of_foreigner"), source="welcome_form"),
                    permanent_address=normalize_vlm_field(vlm_form.get("full_address"), source="welcome_form"),
                    city=normalize_vlm_field(vlm_form.get("city"), source="welcome_form"),
                    country=normalize_vlm_field(vlm_form.get("country"), source="welcome_form"),
                    contact_email=normalize_vlm_field(vlm_form.get("email"), source="welcome_form"),
                    contact_phone=normalize_vlm_field(vlm_form.get("telephone_mobile"), source="welcome_form"),
                    nationality=normalize_vlm_field(vlm_form.get("nationality"), source="welcome_form"),
                    place_of_issue_of_visa=normalize_vlm_field(vlm_form.get("place_of_issue_of_visa"), source="welcome_form"),
                    date_of_arrival_in_india=normalize_vlm_field(vlm_form.get("date_of_arrival_in_india"), source="welcome_form"),
                    arrival_datetime_homestay=normalize_vlm_field(vlm_form.get("datetime_arrival_homestay"), source="welcome_form"),
                    arrived_from=normalize_vlm_field(vlm_form.get("arrived_from"), source="welcome_form"),
                    proceeding_to=normalize_vlm_field(vlm_form.get("proceeding_to"), source="welcome_form"),
                    stay_duration_india=normalize_vlm_field(
                        vlm_form.get("proposed_duration_stay_india") or vlm_form.get("stay_duration_india"),
                        source="welcome_form"
                    ),
                    purpose_of_visit=normalize_vlm_field(vlm_form.get("purpose_of_visit_profession"), source="welcome_form"),
                    expected_departure_date=normalize_vlm_field(vlm_form.get("expected_date_departure"), source="welcome_form"),
                    arrived_country_city_india=normalize_vlm_field(vlm_form.get("country_city_arrived_to_india"), source="welcome_form")
                )

                fused_record.permanent_address = fused_record.documents.welcome_form.permanent_address
                fused_record.city = fused_record.documents.welcome_form.city
                fused_record.country = fused_record.documents.welcome_form.country
                fused_record.contact_email = fused_record.documents.welcome_form.contact_email
                fused_record.contact_phone = fused_record.documents.welcome_form.contact_phone

                form_c_name = fused_record.documents.welcome_form.name_of_foreigner
                if form_c_name.value and (not fused_record.given_names.value and not fused_record.surname.value):
                    parts = form_c_name.value.split(maxsplit=1)
                    fused_record.surname = FieldResult(value=parts[0], confidence=form_c_name.confidence, source="welcome_form")
                    if len(parts) > 1:
                        fused_record.given_names = FieldResult(value=parts[1], confidence=form_c_name.confidence, source="welcome_form")

                if not fused_record.visa_place_of_issue.value:
                    fused_record.visa_place_of_issue = fused_record.documents.welcome_form.place_of_issue_of_visa
                if not fused_record.arrival_date_india.value:
                    fused_record.arrival_date_india = fused_record.documents.welcome_form.date_of_arrival_in_india
                if not fused_record.nationality.value:
                    fused_record.nationality = fused_record.documents.welcome_form.nationality

                fused_record.arrival_datetime_homestay = fused_record.documents.welcome_form.arrival_datetime_homestay
                fused_record.arrived_from = fused_record.documents.welcome_form.arrived_from
                fused_record.proceeding_to = fused_record.documents.welcome_form.proceeding_to
                fused_record.stay_duration_india = fused_record.documents.welcome_form.stay_duration_india
                fused_record.purpose_of_visit = fused_record.documents.welcome_form.purpose_of_visit
                fused_record.expected_departure_date = fused_record.documents.welcome_form.expected_departure_date
                fused_record.arrived_country_city_india = fused_record.documents.welcome_form.arrived_country_city_india

                processing_notes.append("Form-C arrival report (Rows 1-12) extracted via unified single-pass VLM.")
        else:
            logger.info(f"[{job_id}] Stage 4/4: No welcome_form document attached to job. Skipping.")

        total_job_duration = time.perf_counter() - job_start_time
        processing_notes.append(f"Complete pipeline executed in {total_job_duration:.2f}s.")
        fused_record.processing_notes = processing_notes

        save_job_result(job_id, fused_record.model_dump())
        logger.info(f"[+] Job {job_id} successfully completed in {total_job_duration:.2f}s.")

    except Exception as exc:
        total_job_duration = time.perf_counter() - job_start_time
        logger.error(f"[!] Pipeline processing aborted for job {job_id} after {total_job_duration:.2f}s: {str(exc)}", exc_info=True)
        update_job_status(job_id, "FAILED", error_message=f"{str(exc)} (after {total_job_duration:.2f}s)")
    finally:
        enqueued_or_active_jobs.discard(job_id)


# ==============================================================================
# Asynchronous Background Worker Loop
# ==============================================================================

async def background_queue_worker(ml_state: dict) -> None:
    """
    Continuous background loop that waits for jobs added to job_queue
    and processes them sequentially one after another.
    Safely unwraps both UUID string inputs and dictionary records from startup recovery.

    :param ml_state: Global dictionary containing warm Qwen2-VL model handles.
    """
    logger.info("[*] Background async queue worker loop initialized.")
    while True:
        try:
            job_item = await job_queue.get()
            
            if isinstance(job_item, dict):
                job_id = str(job_item.get("job_id", "")).strip()
            else:
                job_id = str(job_item).strip()

            if not job_id:
                logger.warning(f"[*] Queue received invalid job item: {job_item}. Skipping.")
                job_queue.task_done()
                continue

            logger.info(f"[*] Queue dispatched job: {job_id}")

            await process_single_job(job_id, ml_state)

            job_queue.task_done()
        except asyncio.CancelledError:
            logger.info("[*] Background async queue worker task was cancelled.")
            break
        except Exception as exc:
            logger.error(f"[!] Unexpected error in queue worker loop: {str(exc)}", exc_info=True)
            await asyncio.sleep(1.0)