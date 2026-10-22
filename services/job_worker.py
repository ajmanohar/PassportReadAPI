"""
==============================================================================
job_worker.py - PassportReadAPI
==============================================================================
Background worker executing multi-document Form-C extraction pipelines:
Stage 1: Passport Bio-Data (Tesseract MRZ + Qwen2-VL tournament)
Stage 2: Visa / OCI Extraction (Qwen2-VL with explicit document layout anchors)
Stage 3: Arrival Immigration Stamp Extraction (Qwen2-VL entry stamp specialist)
Stage 4: Two-Pass Handwritten Form-C Rule 14 Arrival Report (Qwen2-VL)

Key Improvements:
- Strict MRZ lexical plausibility checks: garbled letters (e.g. CILPIRPIV) are
  downgraded to 0.20 confidence so clean visual VLM reads win the tournament.
- MRZ date normalization from YYMMDD to human YYYY-MM-DD.
- Explicit visual anchor prompts for Indian Visa stickers and immigration entry stamps.
- Programmatic post-extraction sanitization that rejects placeholder hallucinations
  (e.g., '123 Main Street', 'Anytown', '123-456-7890').
- Progressive intermediate persistence to SQLite across all four stages.
==============================================================================
"""

import asyncio
import io
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

from PIL import Image
import torch
from passporteye import read_mrz

# Central configuration and repository imports
from config import settings
from storage.db import update_job_status, save_intermediate_job_result, save_job_result, get_job
from schemas.job import GuestFormCRecord, FieldResult

# Configure structured logging
logger = logging.getLogger("passport_read_api.job_worker")
logger.setLevel(logging.INFO)

# Global in-memory queue holding job UUID strings for sequential execution
job_queue: asyncio.Queue = asyncio.Queue()

# Base disk directory for persisting uploaded document images during asynchronous processing
BASE_STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "jobs")

# Known synthetic placeholder strings to reject programmatically
KNOWN_HALLUCINATIONS = {
    "123 main street",
    "anytown",
    "123-456-7890",
    "sample address",
    "john doe",
    "jane doe",
    "example@domain.com",
    "test@test.com",
    "usa"
}


# ==============================================================================
# Helper Functions: Sanitization, Date Formatting, and Validation
# ==============================================================================

def sanitize_extracted_string(val: Optional[str]) -> Optional[str]:
    """
    Cleans raw string extractions, strips whitespace and quotes, and filters
    out null-equivalent tokens or known synthetic placeholders.

    :param val: Candidate extracted string.
    :return: Clean string or None.
    """
    if val is None:
        return None

    cleaned = str(val).strip().strip("\"'").strip()
    if not cleaned or cleaned.lower() in ["null", "none", "", "n/a", "unknown", "undefined"]:
        return None

    # Reject known synthetic test data
    if cleaned.lower() in KNOWN_HALLUCINATIONS:
        return None

    return cleaned


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

        # Validate calendar month and day ranges
        if mm < 1 or mm > 12 or dd < 1 or dd > 31:
            return cleaned

        current_year = datetime.now().year
        current_century = (current_year // 100) * 100
        current_yy = current_year % 100

        if is_expiry:
            # Expiry dates: usually in current century or next decade
            full_year = current_century + yy if yy >= (current_yy - 10) else (current_century + 100) + yy
        else:
            # Date of birth: past dates
            full_year = current_century + yy if yy <= current_yy else (current_century - 100) + yy

        return f"{full_year:04d}-{mm:02d}-{dd:02d}"
    except Exception:
        return cleaned


def is_plausible_passport_number(number_str: Optional[str]) -> bool:
    """
    Validates whether an extracted passport number follows standard ICAO 9303 conventions.
    Rejects strings with only letters (such as OCR hallucination 'CILPIRPIV').

    :param number_str: Cleaned passport number string.
    :return: True if plausible, False otherwise.
    """
    if not number_str:
        return False

    cleaned = re.sub(r"[^A-Za-z0-9]", "", number_str.upper())
    # Standard passports are 7 to 10 characters and almost always contain at least 2 digits
    if len(cleaned) < 6 or len(cleaned) > 12:
        return False

    digit_count = sum(1 for c in cleaned if c.isdigit())
    if digit_count < 2:
        # Reject letter-only or near-letter-only garbled reads
        return False

    return True


# ==============================================================================
# Image Preparation Helper
# ==============================================================================

def load_and_preprocess_image(file_path: str, max_dimension: Optional[int] = None) -> Optional[Image.Image]:
    """
    Loads an image from disk storage, standardizes color channels to RGB,
    and downscales proportionally if the maximum dimension exceeds configured limits.

    :param file_path: Absolute or relative filesystem path to the image file.
    :param max_dimension: Maximum allowed width or height in pixels.
    :return: Prepared PIL Image instance, or None if the file cannot be read.
    """
    if not os.path.exists(file_path):
        logger.warning(f"[ImageLoader] File not found on disk: {file_path}")
        return None

    target_max_dim = max_dimension if max_dimension is not None else settings.max_image_dimension

    try:
        start_t = time.perf_counter()
        image = Image.open(file_path)
        orig_w, orig_h = image.size

        # Standardize color profile to 3-channel RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Downscale proportionally to bound attention matrix computation
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

def run_vlm_inference(image: Image.Image, system_prompt: str, ml_state: dict, stage_label: str = "VLM") -> dict:
    """
    Executes a structured query against the shared in-memory Qwen2-VL model instance
    with detailed per-stage execution timing and parses the output into a Python dictionary.

    :param image: Preprocessed PIL Image instance.
    :param system_prompt: Guidance text instructing the VLM on expected JSON fields.
    :param ml_state: Shared dictionary holding 'model' and 'processor' instances.
    :param stage_label: Diagnostic label for logging.
    :return: Extracted key-value dictionary parsed from the model's generated JSON.
    """
    from qwen_vl_utils import process_vision_info

    processor = ml_state.get("processor")
    model = ml_state.get("model")

    if not model or not processor:
        raise RuntimeError("Qwen2-VL model and processor are not loaded in ml_state.")

    torch.set_num_threads(settings.torch_cpu_threads)
    start_vlm_time = time.perf_counter()
    logger.info(f"[{stage_label}] Preparing prompt tensor inputs (image: {image.size[0]}x{image.size[1]})...")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": settings.min_pixels,
                    "max_pixels": settings.max_pixels
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

    logger.info(f"[{stage_label}] Beginning PyTorch CPU token generation (max_new_tokens={settings.max_new_tokens})...")
    gen_start = time.perf_counter()

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
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
            return json.loads(match.group(0))
        return json.loads(raw_text)
    except Exception as parse_err:
        logger.warning(
            f"[{stage_label}] Failed to parse structured JSON: {parse_err}. "
            f"Raw excerpt: {raw_text[:250]}"
        )
        return {"confidence_note": f"Raw model response: {raw_text}"}


# ==============================================================================
# Tournament Selection & Normalization Helpers
# ==============================================================================

def normalize_vlm_field(raw_field: Any, default_score: float = 0.85, source: str = "qwen2_vl") -> FieldResult:
    """
    Parses a nested or scalar field returned by Qwen2-VL into a validated FieldResult.

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
    if not clean_val:
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
    Conducts a head-to-head confidence tournament for a specific field between
    Tesseract MRZ and Qwen2-VL, selecting the candidate with higher confidence.

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
    Queries SQLite and inspects the on-disk job directory to discover any files
    attached after the initial job ingestion.

    :param job_id: UUID tracking string.
    :return: Fresh mapping of document types to verified local file paths.
    """
    job_record = get_job(job_id)
    doc_paths: Dict[str, str] = job_record.get("doc_paths", {}) if job_record else {}

    job_dir = os.path.join(BASE_STORAGE_DIR, job_id)
    if os.path.isdir(job_dir):
        for filename in os.listdir(job_dir):
            full_path = os.path.join(job_dir, filename)
            if not os.path.isfile(full_path):
                continue
            if filename.startswith("bio_") and "passport_bio" not in doc_paths:
                doc_paths["passport_bio"] = full_path
            elif filename.startswith("visa_") and "visa_page" not in doc_paths:
                doc_paths["visa_page"] = full_path
            elif filename.startswith("stamp_") and "entry_stamp" not in doc_paths:
                doc_paths["entry_stamp"] = full_path
            elif filename.startswith("form_") and "welcome_form" not in doc_paths:
                doc_paths["welcome_form"] = full_path

    return doc_paths


# ==============================================================================
# Job Processing Pipeline
# ==============================================================================

async def process_single_job(job_id: str, ml_state: dict) -> None:
    """
    Executes the multi-document extraction pipeline for a single queued job.
    Runs Tesseract MRZ and Qwen2-VL for the bio page, selects the
    best candidate per field, and continues through Visa, Stamp, and Welcome Form.

    :param job_id: Unique UUID string identifier of the target job.
    :param ml_state: Shared dictionary holding model handles.
    """
    job_start_time = time.perf_counter()
    logger.info(f"[*] Starting extraction pipeline for job: {job_id}")
    update_job_status(job_id, "PROCESSING")

    fused_record = GuestFormCRecord()
    processing_notes = []

    try:
        # ======================================================================
        # 1. PASSPORT BIO PAGE: DUAL-ENGINE TOURNAMENT (MRZ + VLM)
        # ======================================================================
        doc_paths = refresh_document_paths(job_id)
        bio_path = doc_paths.get("passport_bio")

        if bio_path and os.path.exists(bio_path):
            logger.info(f"[{job_id}] Stage 1/4: Dual-engine extraction for passport bio page ({bio_path})...")

            # --- Engine A: Tesseract MRZ Execution ---
            mrz_start = time.perf_counter()
            mrz_candidates: Dict[str, FieldResult] = {}
            valid_mrz_flag = False

            try:
                with open(bio_path, "rb") as f:
                    image_bytes = f.read()

                mrz_record = await asyncio.to_thread(read_mrz, io.BytesIO(image_bytes))

                if mrz_record is not None:
                    mrz_dict = mrz_record.to_dict() if hasattr(mrz_record, "to_dict") else {}
                    score_val = getattr(mrz_record, "valid_score", 0)
                    valid_score = int(score_val) if isinstance(score_val, (int, float)) else 0
                    base_text_conf = round(max(0.0, min(1.0, valid_score / 100.0)), 2)

                    check_number = getattr(mrz_record, "check_number", False)
                    check_dob = getattr(mrz_record, "check_date_of_birth", False)
                    check_exp = getattr(mrz_record, "check_expiration_date", False)
                    check_composite = getattr(mrz_record, "check_composite", False)

                    valid_mrz_flag = bool(check_composite or valid_score >= 70)

                    def mk_checked(val: Optional[str], check_passed: bool, is_passport_num: bool = False, is_date: bool = False, is_expiry: bool = False) -> FieldResult:
                        clean = sanitize_extracted_string(val)
                        if not clean:
                            return FieldResult(value=None, confidence=0.0, source="tesseract_mrz")

                        # Validate passport number plausibility
                        if is_passport_num:
                            if not is_plausible_passport_number(clean):
                                # If garbled (like CILPIRPIV), drop confidence to 0.20 so VLM wins
                                return FieldResult(value=clean, confidence=0.20, source="tesseract_mrz")
                            return FieldResult(value=clean, confidence=0.95 if check_passed else 0.40, source="tesseract_mrz")

                        # Normalize 6-digit MRZ date into YYYY-MM-DD
                        if is_date:
                            formatted_date = parse_mrz_date(clean, is_expiry=is_expiry)
                            return FieldResult(value=formatted_date, confidence=0.95 if check_passed else 0.40, source="tesseract_mrz")

                        return FieldResult(
                            value=clean,
                            confidence=0.95 if check_passed else 0.40,
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
                    mrz_candidates["passport_number"] = mk_checked(mrz_dict.get("number"), check_number, is_passport_num=True)
                    mrz_candidates["nationality"] = mk_text(mrz_dict.get("nationality"))
                    mrz_candidates["date_of_birth"] = mk_checked(mrz_dict.get("date_of_birth"), check_dob, is_date=True, is_expiry=False)
                    mrz_candidates["sex"] = mk_text(mrz_dict.get("sex"))
                    mrz_candidates["passport_expiry_date"] = mk_checked(mrz_dict.get("expiration_date"), check_exp, is_date=True, is_expiry=True)

                    mrz_duration = time.perf_counter() - mrz_start
                    logger.info(
                        f"[{job_id}] Tesseract MRZ completed in {mrz_duration:.2f}s "
                        f"(valid={valid_mrz_flag}, score={valid_score})."
                    )
                    processing_notes.append(f"MRZ parsed in {mrz_duration:.2f}s (valid: {valid_mrz_flag}, score: {valid_score}).")
                else:
                    mrz_duration = time.perf_counter() - mrz_start
                    logger.info(f"[{job_id}] Tesseract MRZ found no patterns ({mrz_duration:.2f}s).")
                    processing_notes.append(f"MRZ not detected ({mrz_duration:.2f}s).")
            except Exception as mrz_exc:
                logger.warning(f"[{job_id}] MRZ extraction error: {str(mrz_exc)}", exc_info=True)
                processing_notes.append(f"MRZ error: {str(mrz_exc)}")

            # --- Engine B: Qwen2-VL Visual Bio Execution ---
            vlm_candidates: Dict[str, FieldResult] = {}
            bio_image = await asyncio.to_thread(load_and_preprocess_image, bio_path)

            if bio_image:
                bio_prompt = (
                    "Carefully examine the visual photo page of the passport (Visual Inspection Zone).\n"
                    "Extract the printed text into a valid JSON object matching this schema:\n"
                    "{\n"
                    '  "surname": {"value": string or null, "confidence": 0.85},\n'
                    '  "given_names": {"value": string or null, "confidence": 0.85},\n'
                    '  "passport_number": {"value": string or null, "confidence": 0.85},\n'
                    '  "nationality": {"value": string or null, "confidence": 0.85},\n'
                    '  "date_of_birth": {"value": string or null, "confidence": 0.85},\n'
                    '  "sex": {"value": "M" or "F" or null, "confidence": 0.85},\n'
                    '  "expiry_date": {"value": string or null, "confidence": 0.85}\n'
                    "}\n"
                    "Instructions:\n"
                    "- Read the human-readable text printed near the labels 'Surname', 'Given Names', 'Passport No', 'Date of Birth'.\n"
                    "- For passport_number, transcribe the exact alphanumeric string (e.g. C9K82159L).\n"
                    "- If unreadable, return null for value.\n"
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

            # --- Engine Tournament: Select Best Field for Every Bio Attribute ---
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

            # Persist Stage 1 intermediate progress to SQLite immediately
            fused_record.processing_notes = list(processing_notes)
            save_intermediate_job_result(job_id, fused_record.model_dump())
            logger.info(f"[{job_id}] Stage 1/4 completed and intermediate state persisted to database.")
        else:
            logger.warning(f"[{job_id}] Passport bio image not found on disk: {bio_path}")
            processing_notes.append("Passport bio image missing from job directory.")

        # ======================================================================
        # 2. VISA / OCI EXTRACTION (QWEN2-VL)
        # ======================================================================
        doc_paths = refresh_document_paths(job_id)
        visa_path = doc_paths.get("visa_page")

        if visa_path and os.path.exists(visa_path):
            logger.info(f"[{job_id}] Stage 2/4: Running Qwen2-VL visa extraction ({visa_path})...")
            visa_image = await asyncio.to_thread(load_and_preprocess_image, visa_path)
            if visa_image:
                visa_prompt = (
                    "Examine this Indian Visa sticker or OCI card document image carefully.\n"
                    "Locate the printed fields and extract them into a valid JSON object matching this schema:\n"
                    "{\n"
                    '  "visa_number": {"value": string or null, "confidence": 0.85},\n'
                    '  "visa_type": {"value": string or null, "confidence": 0.85},\n'
                    '  "issue_date": {"value": string or null, "confidence": 0.85},\n'
                    '  "expiry_date": {"value": string or null, "confidence": 0.85},\n'
                    '  "place_of_issue": {"value": string or null, "confidence": 0.85}\n'
                    "}\n"
                    "Visual Layout Anchors:\n"
                    "- 'visa_number': Look for 'Visa No', 'Document No', or a bold 7-9 digit printed number (often printed in red or black at the top right or middle).\n"
                    "- 'visa_type': Look for 'Type', 'Class', or category such as 'TOURIST', 'T', 'BUSINESS', 'X', 'STUDENT', 'OCI'.\n"
                    "- 'issue_date': The date of issue printed near 'Date of Issue' or 'Issued On'.\n"
                    "- 'expiry_date': The date printed near 'Date of Expiry' or 'Valid Until'.\n"
                    "- 'place_of_issue': The authority or consular post (e.g., 'BUREAU OF IMMIGRATION', 'BERLIN', 'LONDON', 'NEW YORK').\n"
                    "- If a field is not present or illegible, return null for value.\n"
                    "- Output JSON only."
                )
                vlm_visa = await asyncio.to_thread(run_vlm_inference, visa_image, visa_prompt, ml_state, "Visa VLM")
                fused_record.visa_number = normalize_vlm_field(vlm_visa.get("visa_number"))
                fused_record.visa_type = normalize_vlm_field(vlm_visa.get("visa_type"))
                fused_record.visa_issue_date = normalize_vlm_field(vlm_visa.get("issue_date"))
                fused_record.visa_expiry_date = normalize_vlm_field(vlm_visa.get("expiry_date"))
                fused_record.visa_place_of_issue = normalize_vlm_field(vlm_visa.get("place_of_issue"))
                processing_notes.append("Visa / OCI details extracted via Qwen2-VL.")

                # Persist Stage 2 intermediate progress to SQLite immediately
                fused_record.processing_notes = list(processing_notes)
                save_intermediate_job_result(job_id, fused_record.model_dump())
                logger.info(f"[{job_id}] Stage 2/4 completed and intermediate state persisted to database.")
        else:
            logger.info(f"[{job_id}] Stage 2/4: No visa_page document attached to job. Skipping.")

        # ======================================================================
        # 3. IMMIGRATION ENTRY STAMP EXTRACTION (QWEN2-VL)
        # ======================================================================
        doc_paths = refresh_document_paths(job_id)
        stamp_path = doc_paths.get("entry_stamp")

        if stamp_path and os.path.exists(stamp_path):
            logger.info(f"[{job_id}] Stage 3/4: Running Qwen2-VL entry stamp extraction ({stamp_path})...")
            stamp_image = await asyncio.to_thread(load_and_preprocess_image, stamp_path)
            if stamp_image:
                stamp_prompt = (
                    "Examine this passport stamp page. Find the Indian immigration ARRIVAL or ENTRY stamp (usually rectangular or oval with an airplane or 'INDIAN IMMIGRATION').\n"
                    "Ignore departure or exit stamps.\n"
                    "Extract details into this JSON schema:\n"
                    "{\n"
                    '  "arrival_date": {"value": string or null, "confidence": 0.85},\n'
                    '  "arrival_port": {"value": string or null, "confidence": 0.85}\n'
                    "}\n"
                    "Visual Layout Anchors:\n"
                    "- 'arrival_date': Look for the date stamped in the center (e.g., '06 OCT 2026' or '2023-04-01').\n"
                    "- 'arrival_port': Read the name of the airport or seaport printed along the top or bottom border (e.g. 'COCHIN AIRPORT', 'TRIVANDRUM INTL AIRPORT', 'DELHI', 'MUMBAI').\n"
                    "- If unreadable, return null for value.\n"
                    "- Output JSON only."
                )
                vlm_stamp = await asyncio.to_thread(run_vlm_inference, stamp_image, stamp_prompt, ml_state, "Stamp VLM")
                fused_record.arrival_date_india = normalize_vlm_field(vlm_stamp.get("arrival_date"))
                fused_record.arrival_port_india = normalize_vlm_field(vlm_stamp.get("arrival_port"))
                processing_notes.append("Entry stamp details extracted via Qwen2-VL.")

                # Persist Stage 3 intermediate progress to SQLite immediately
                fused_record.processing_notes = list(processing_notes)
                save_intermediate_job_result(job_id, fused_record.model_dump())
                logger.info(f"[{job_id}] Stage 3/4 completed and intermediate state persisted to database.")
        else:
            logger.info(f"[{job_id}] Stage 3/4: No entry_stamp document attached to job. Skipping.")

        # ======================================================================
        # 4. HANDWRITTEN ARRIVAL REPORT EXTRACTION (FORM-C RULE 14)
        # ======================================================================
        doc_paths = refresh_document_paths(job_id)
        welcome_path = doc_paths.get("welcome_form")

        if welcome_path and os.path.exists(welcome_path):
            logger.info(f"[{job_id}] Stage 4/4: Running Qwen2-VL handwritten Form-C extraction ({welcome_path})...")
            welcome_image = await asyncio.to_thread(load_and_preprocess_image, welcome_path)
            if welcome_image:
                welcome_prompt = (
                    "You are an administrative document assistant digitizing an Indian Form C (Rule 14) Arrival Report homestay registration card.\n"
                    "Transcribe the genuine handwriting written by the foreigner guest in the numbered rows:\n"
                    "- Row 1: Name of foreigner (Surname First)\n"
                    "- Row 2: Permanent residential address in their home country, City, Country, E-mail, Mobile Phone\n"
                    "- Row 3: Nationality\n"
                    "- Row 4: Place of issue of visa\n"
                    "- Row 5: Date of Arrival in India\n"
                    "- Row 6: Date and Time of Arrival in homestay\n"
                    "- Row 7: Arrived from (city or state)\n"
                    "- Row 8: Proceeding to\n"
                    "- Row 9: Proposed duration of stay in India\n"
                    "- Row 10: Purpose of visit & Profession\n"
                    "- Row 11: Expected date of departure\n"
                    "- Row 12: Country & city arrived to India\n\n"
                    "CRITICAL INSTRUCTIONS:\n"
                    "- Do NOT invent or output fake example data like '123 Main Street', 'Anytown', or '123-456-7890'.\n"
                    "- If a field is blank or illegible, you MUST return null for value.\n"
                    "- Transcribe only genuine handwriting.\n"
                    "Return ONLY valid JSON matching this structure:\n"
                    "{\n"
                    '  "full_address": {"value": string or null, "confidence": 0.80},\n'
                    '  "city": {"value": string or null, "confidence": 0.80},\n'
                    '  "country": {"value": string or null, "confidence": 0.80},\n'
                    '  "email": {"value": string or null, "confidence": 0.80},\n'
                    '  "telephone_mobile": {"value": string or null, "confidence": 0.80},\n'
                    '  "nationality": {"value": string or null, "confidence": 0.80},\n'
                    '  "place_of_issue_of_visa": {"value": string or null, "confidence": 0.80},\n'
                    '  "date_of_arrival_in_india": {"value": string or null, "confidence": 0.80},\n'
                    '  "datetime_arrival_homestay": {"value": string or null, "confidence": 0.80},\n'
                    '  "arrived_from": {"value": string or null, "confidence": 0.80},\n'
                    '  "proceeding_to": {"value": string or null, "confidence": 0.80},\n'
                    '  "proposed_duration_stay_india": {"value": string or null, "confidence": 0.80},\n'
                    '  "purpose_of_visit_profession": {"value": string or null, "confidence": 0.80},\n'
                    '  "expected_date_departure": {"value": string or null, "confidence": 0.80},\n'
                    '  "country_city_arrived_to_india": {"value": string or null, "confidence": 0.80}\n'
                    "}"
                )
                vlm_form = await asyncio.to_thread(run_vlm_inference, welcome_image, welcome_prompt, ml_state, "Welcome Form VLM")

                # Map extracted Form-C items into fused record with source attribution
                fused_record.permanent_address = normalize_vlm_field(vlm_form.get("full_address"), source="welcome_form")
                fused_record.city = normalize_vlm_field(vlm_form.get("city"), source="welcome_form")
                fused_record.country = normalize_vlm_field(vlm_form.get("country"), source="welcome_form")
                fused_record.contact_email = normalize_vlm_field(vlm_form.get("email"), source="welcome_form")
                fused_record.contact_phone = normalize_vlm_field(vlm_form.get("telephone_mobile"), source="welcome_form")
                fused_record.arrival_datetime_homestay = normalize_vlm_field(vlm_form.get("datetime_arrival_homestay"), source="welcome_form")
                fused_record.arrived_from = normalize_vlm_field(vlm_form.get("arrived_from"), source="welcome_form")
                fused_record.proceeding_to = normalize_vlm_field(vlm_form.get("proceeding_to"), source="welcome_form")
                fused_record.stay_duration_india = normalize_vlm_field(vlm_form.get("proposed_duration_stay_india"), source="welcome_form")
                fused_record.purpose_of_visit = normalize_vlm_field(vlm_form.get("purpose_of_visit_profession"), source="welcome_form")
                fused_record.expected_departure_date = normalize_vlm_field(vlm_form.get("expected_date_departure"), source="welcome_form")
                fused_record.arrived_country_city_india = normalize_vlm_field(vlm_form.get("country_city_arrived_to_india"), source="welcome_form")

                # Fallback comparisons if passport or visa did not detect values
                if not fused_record.visa_place_of_issue.value:
                    fused_record.visa_place_of_issue = normalize_vlm_field(vlm_form.get("place_of_issue_of_visa"), source="welcome_form")
                if not fused_record.arrival_date_india.value:
                    fused_record.arrival_date_india = normalize_vlm_field(vlm_form.get("date_of_arrival_in_india"), source="welcome_form")
                if not fused_record.nationality.value:
                    fused_record.nationality = normalize_vlm_field(vlm_form.get("nationality"), source="welcome_form")

                processing_notes.append("Handwritten Form-C arrival report extracted via Qwen2-VL.")
        else:
            logger.info(f"[{job_id}] Stage 4/4: No welcome_form document attached to job. Skipping.")

        # Persist final fused record with diagnostic tournament logs and total elapsed duration
        total_job_duration = time.perf_counter() - job_start_time
        processing_notes.append(f"Complete pipeline executed in {total_job_duration:.2f}s.")
        fused_record.processing_notes = processing_notes

        # Transition status to COMPLETED and commit final result
        save_job_result(job_id, fused_record.model_dump())
        logger.info(f"[+] Job {job_id} successfully completed in {total_job_duration:.2f}s.")

    except Exception as exc:
        total_job_duration = time.perf_counter() - job_start_time
        logger.error(f"[!] Pipeline processing aborted for job {job_id} after {total_job_duration:.2f}s: {str(exc)}", exc_info=True)
        update_job_status(job_id, "FAILED", error_message=f"{str(exc)} (after {total_job_duration:.2f}s)")


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
            job_id = await job_queue.get()
            logger.info(f"[*] Queue dispatched job: {job_id}")

            await process_single_job(job_id, ml_state)

            job_queue.task_done()
        except asyncio.CancelledError:
            logger.info("[*] Background async queue worker task was cancelled.")
            break
        except Exception as exc:
            logger.error(f"[!] Unexpected error in queue worker loop: {str(exc)}", exc_info=True)
            await asyncio.sleep(1.0)