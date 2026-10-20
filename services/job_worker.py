# ==============================================================================
# job_worker.py - PassportReadAPI
# ==============================================================================
# Background batch execution pipeline and serial queue worker for asynchronous Form-C jobs.
# Processes multi-document bundles (Passport Bio, Visa, Entry Stamp, Welcome Form) sequentially.
# Executes an unconditional 'best-of-both' confidence tournament between Tesseract MRZ and
# Qwen2-VL for the Passport Bio page, selecting the highest-confidence extraction per field.
#
# Server-Side Document Normalization:
# - Applies EXIF transposition to correct camera orientation tags.
# - High-confidence OSD thresholding: Ignores low-confidence (< 2.0) or non-Latin OSD false positives.
# - Neutral administrative archival prompts preventing safety refusal triggers.
# - Explicit header separation: Distinguishes pre-printed homestay letterhead from guest residential address.
# - Dynamic visual token bounding respecting per-stage patch ceilings.
# - High-definition visual patch allocation for fine cursive handwriting.
# - OCR-Assisted Grounding: Supplies full localized Tesseract tokens as context hints to Qwen2-VL.
# - Row-Anchored Form-C Slices: Explicitly anchors rows 1-5 (Pass 4A) and rows 6-12 (Pass 4B).
# - Resilient Queue Ingestion: Safely handles both raw UUID strings and startup recovery dicts.
# ==============================================================================

import asyncio
import io
import json
import logging
import os
import re
import time
from typing import Dict, Any, Optional, List, Union

from PIL import Image, ImageOps
import pytesseract
import torch
from passporteye import read_mrz

# Central configuration and repository imports
from config import settings
from storage.db import update_job_status, save_intermediate_job_result, save_job_result, get_job
from schemas.job import GuestFormCRecord, FieldResult

# Set up module-level logger
logger = logging.getLogger("job_worker")
logger.setLevel(logging.INFO)

# Global in-memory queue holding job tracking identifiers for sequential execution by the worker loop
job_queue: asyncio.Queue = asyncio.Queue()

# Base disk directory for persisting uploaded document images during asynchronous processing
BASE_STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "jobs")

# Blacklist of known synthetic placeholder tokens to reject at field normalization
FORBIDDEN_HALLUCINATIONS: List[str] = [
    "parthasarthy",
    "parthasarathy",
    "example@domain.com",
    "9876543210",
    "dummy",
    "lorem ipsum",
    "john.doe"
]


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
    
    val_clean = str(val).strip()
    if not val_clean or val_clean.lower() in ["null", "none", "n/a", "unknown", "", "nil"]:
        return None
    
    val_lower = val_clean.lower()
    for bad_token in FORBIDDEN_HALLUCINATIONS:
        if bad_token in val_lower:
            logger.warning(f"[Sanitizer] Blocked synthetic placeholder string containing '{bad_token}': {val_clean}")
            return None
            
    return val_clean


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

    # Resolve default pixel ceiling from application settings
    target_max_dim = max_dimension if max_dimension is not None else settings.max_image_dimension

    try:
        start_t = time.perf_counter()
        raw_img = Image.open(file_path)

        # 1. Standardize orientation using EXIF orientation tags (if present)
        image = ImageOps.exif_transpose(raw_img)

        # 2. Standardize color profile to 3-channel RGB (eliminates alpha channel or CMYK artifacts)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 3. Guarded auto-orientation check for documents
        # Only rotate if Tesseract OSD is highly confident (conf >= 2.0) and script is Latin.
        # This prevents low-confidence false rotations (e.g. conf 0.47 classifying Latin handwriting as Japanese).
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

                # High-confidence threshold guardrail
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

    # Resolve visual patch boundaries dynamically
    effective_min_pixels = min_pixels_override if min_pixels_override is not None else settings.min_pixels
    effective_max_pixels = max_pixels_override if max_pixels_override is not None else settings.max_pixels

    start_vlm_time = time.perf_counter()
    logger.info(
        f"[{stage_label}] Preparing prompt tensor inputs (image: {image.size[0]}x{image.size[1]}, "
        f"patches: {effective_min_pixels}-{effective_max_pixels}px)..."
    )

    # Format user prompt according to Qwen2-VL chat conversation template
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

    effective_max_tokens = max_tokens_override if max_tokens_override is not None else settings.max_new_tokens
    logger.info(f"[{stage_label}] Beginning PyTorch CPU token generation (max_new_tokens={effective_max_tokens})...")
    gen_start = time.perf_counter()

    # Deterministic greedy decoding (do_sample=False) to suppress probabilistic hallucinations
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=effective_max_tokens,
            do_sample=False
        )

    gen_duration = time.perf_counter() - gen_start
    total_duration = time.perf_counter() - start_vlm_time

    # Separate model-generated response tokens from the input prompt tokens
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    tokens_generated = len(generated_ids_trimmed[0]) if generated_ids_trimmed else 0

    # Decode tokens into UTF-8 string
    raw_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )[0]

    logger.info(
        f"[{stage_label}] Generated {tokens_generated} tokens in {gen_duration:.2f}s "
        f"({tokens_generated / max(0.01, gen_duration):.1f} tok/s). Total stage latency: {total_duration:.2f}s."
    )

    # Extract JSON string payload bounded by braces with syntax cleanup fallback
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

    # Sanitize string through anti-hallucination filter
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

    :param job_input: Either a UUID string or a recovered dictionary holding 'job_id'.
    :param ml_state: Shared dictionary holding model handles.
    """
    # Defensive unwrapping: Extract UUID string if a dictionary was passed from startup recovery
    if isinstance(job_input, dict):
        job_id = str(job_input.get("job_id", "")).strip()
    else:
        job_id = str(job_input).strip()

    if not job_id:
        logger.error("[!] Dispatched empty or invalid job identifier to process_single_job.")
        return

    job_start_time = time.perf_counter()
    logger.info(f"[*] Starting extraction pipeline for job: {job_id}")
    update_job_status(job_id, "PROCESSING")

    fused_record = GuestFormCRecord()
    processing_notes: List[str] = []

    try:
        # ======================================================================
        # 1. PASSPORT BIO PAGE: DUAL-ENGINE TOURNAMENT (MRZ + VLM)
        # ======================================================================
        doc_paths = refresh_document_paths(job_id)
        bio_path = doc_paths.get("passport_bio")

        if bio_path and os.path.exists(bio_path):
            logger.info(f"[{job_id}] Stage 1/4: Dual-engine extraction for passport bio page ({bio_path})...")

            # --- Engine A: Tesseract MRZ ---
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

                    valid_mrz_flag = bool(check_composite or valid_score >= 80)

                    def mk_checked(val: Optional[str], check_passed: bool) -> FieldResult:
                        clean = sanitize_extracted_string(val)
                        if not clean:
                            return FieldResult(value=None, confidence=0.0, source="tesseract_mrz")
                        return FieldResult(
                            value=clean,
                            confidence=0.99 if check_passed else 0.30,
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
                    mrz_candidates["date_of_birth"] = mk_checked(mrz_dict.get("date_of_birth"), check_dob)
                    mrz_candidates["sex"] = mk_text(mrz_dict.get("sex"))
                    mrz_candidates["passport_expiry_date"] = mk_checked(mrz_dict.get("expiration_date"), check_exp)

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

            # --- Engine B: Qwen2-VL Visual Bio ---
            vlm_candidates: Dict[str, FieldResult] = {}
            bio_image = await asyncio.to_thread(load_and_preprocess_image, bio_path)

            if bio_image:
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
                    "You are an administrative document verification assistant inspecting an Indian Visa sticker or OCI card document image.\n"
                    "Extract only the text physically printed on the document into a valid JSON object matching this schema:\n"
                    "{\n"
                    '  "visa_number": {"value": string or null, "confidence": float},\n'
                    '  "visa_type": {"value": string or null, "confidence": float},\n'
                    '  "issue_date": {"value": string or null, "confidence": float},\n'
                    '  "expiry_date": {"value": string or null, "confidence": float},\n'
                    '  "place_of_issue": {"value": string or null, "confidence": float}\n'
                    "}\n"
                    "Instructions:\n"
                    "- If an expiry date is not printed on the document, return null for expiry_date. Do not calculate or assume an expiry date based on the issue date.\n"
                    "- For place_of_issue, transcribe the exact issuing mission or authority printed on the document. If not printed or illegible, return null.\n"
                    "- If any field is missing or unreadable, return null for value and 0.0 for confidence.\n"
                    "- Output JSON only."
                )
                vlm_visa = await asyncio.to_thread(run_vlm_inference, visa_image, visa_prompt, ml_state, "Visa VLM")
                fused_record.visa_number = normalize_vlm_field(vlm_visa.get("visa_number"))
                fused_record.visa_type = normalize_vlm_field(vlm_visa.get("visa_type"))
                fused_record.visa_issue_date = normalize_vlm_field(vlm_visa.get("issue_date"))
                fused_record.visa_expiry_date = normalize_vlm_field(vlm_visa.get("expiry_date"))
                fused_record.visa_place_of_issue = normalize_vlm_field(vlm_visa.get("place_of_issue"))
                processing_notes.append("Visa / OCI details extracted via Qwen2-VL.")

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
                    "You are an administrative document verification assistant inspecting an Indian immigration entry/arrival rubber stamp image.\n"
                    "Extract the arrival details into a valid JSON object matching this schema:\n"
                    "{\n"
                    '  "arrival_date": {"value": string or null, "confidence": float},\n'
                    '  "arrival_port": {"value": string or null, "confidence": float}\n'
                    "}\n"
                    "Instructions:\n"
                    "- Read only ARRIVAL or ENTRY stamps. Ignore departure or exit stamps.\n"
                    "- Transcribe the exact port or airport name printed on the arrival stamp. If cut off, smudged, or illegible, return null for value and 0.0 for confidence.\n"
                    "- If the arrival date is illegible or not present, return null for value and 0.0 for confidence.\n"
                    "- Output JSON only."
                )
                vlm_stamp = await asyncio.to_thread(run_vlm_inference, stamp_image, stamp_prompt, ml_state, "Stamp VLM")
                fused_record.arrival_date_india = normalize_vlm_field(vlm_stamp.get("arrival_date"))
                fused_record.arrival_port_india = normalize_vlm_field(vlm_stamp.get("arrival_port"))
                processing_notes.append("Entry stamp details extracted via Qwen2-VL.")

                fused_record.processing_notes = list(processing_notes)
                save_intermediate_job_result(job_id, fused_record.model_dump())
                logger.info(f"[{job_id}] Stage 3/4 completed and intermediate state persisted to database.")
        else:
            logger.info(f"[{job_id}] Stage 3/4: No entry_stamp document attached to job. Skipping.")

        # ======================================================================
        # 4. HANDWRITTEN FORM-C ARRIVAL REPORT (OCR-GROUNDED TWO-PASS EXTRACTION)
        # ======================================================================
        doc_paths = refresh_document_paths(job_id)
        welcome_path = doc_paths.get("welcome_form")

        if welcome_path and os.path.exists(welcome_path):
            logger.info(f"[{job_id}] Stage 4/4: Initializing OCR-grounded Form-C extraction ({welcome_path})...")
            welcome_raw = await asyncio.to_thread(load_and_preprocess_image, welcome_path)
            
            if welcome_raw:
                img_w, img_h = welcome_raw.size

                # ------------------------------------------------------------------
                # Pass 4A: Top Section (Rows 1 to 5: Identity, Residence, Visa, Arrival)
                # Crop top 52% of the document
                # ------------------------------------------------------------------
                top_crop_box = (0, 0, img_w, int(img_h * 0.52))
                welcome_top_img = welcome_raw.crop(top_crop_box)
                logger.info(f"[{job_id}] Pass 4A: Cropped top section ({img_w}x{welcome_top_img.size[1]}).")

                # Run fast local OCR specifically on the top crop to get ground-truth tokens
                ocr_top_text = ""
                try:
                    ocr_top_raw = await asyncio.to_thread(pytesseract.image_to_string, welcome_top_img)
                    ocr_top_lines = [line.strip() for line in ocr_top_raw.splitlines() if line.strip()]
                    ocr_top_text = "\n".join(ocr_top_lines)
                except Exception as e:
                    ocr_top_text = "No OCR hints available."

                welcome_top_prompt = (
                    "You are an administrative office assistant digitizing the top section (Rows 1 to 5) of an official Form C (Rule 14) Arrival Report homestay registration card.\n"
                    "CRITICAL: The card contains pre-printed header letterhead at the top ('HOMESTED, 1386 A, PARVATHY NILAYAM, THAMARAKULAM ROAD, KOCHI'). IGNORE that homestay header completely.\n"
                    "Extract ONLY the foreigner guest's handwritten entries in Rows 1 to 5.\n\n"
                    "Here are raw OCR character tokens detected on this top section:\n"
                    f"\"\"\"\n{ocr_top_text}\n\"\"\"\n\n"
                    "Examine the numbered rows on the card:\n"
                    "- Row 1: Name of foreigner (Surname First)\n"
                    "- Row 2: Full Address (handwritten residential street/town in their home country), City, Country, E-mail, Telephone / Mobile\n"
                    "- Row 3: Nationality\n"
                    "- Row 4: Place of issue of visa\n"
                    "- Row 5: Date of Arrival in India\n\n"
                    "Transcribe the genuine handwriting into this JSON schema:\n"
                    "{\n"
                    '  "name_of_foreigner": {"value": string or null, "confidence": float},\n'
                    '  "full_address": {"value": string or null, "confidence": float},\n'
                    '  "city": {"value": string or null, "confidence": float},\n'
                    '  "country": {"value": string or null, "confidence": float},\n'
                    '  "email": {"value": string or null, "confidence": float},\n'
                    '  "telephone_mobile": {"value": string or null, "confidence": float},\n'
                    '  "nationality": {"value": string or null, "confidence": float},\n'
                    '  "place_of_issue_of_visa": {"value": string or null, "confidence": float},\n'
                    '  "date_of_arrival_in_india": {"value": "YYYY-MM-DD" or null, "confidence": float}\n'
                    "}\n"
                    "Instructions:\n"
                    "- full_address, city, and country MUST be the foreigner's permanent home address written by hand, NOT the Indian homestay address.\n"
                    "- If a line is blank or empty, return null for value and 0.0 for confidence.\n"
                    "- Output JSON only."
                )

                # Allocate high-definition visual tokens for crisp handwriting reading
                vlm_form_top = await asyncio.to_thread(
                    run_vlm_inference,
                    welcome_top_img,
                    welcome_top_prompt,
                    ml_state,
                    "Welcome Form (Top Section) VLM",
                    384,
                    65536,
                    262144
                )

                fused_record.permanent_address = normalize_vlm_field(vlm_form_top.get("full_address"), source="welcome_form")
                fused_record.city = normalize_vlm_field(vlm_form_top.get("city"), source="welcome_form")
                fused_record.country = normalize_vlm_field(vlm_form_top.get("country"), source="welcome_form")
                fused_record.contact_email = normalize_vlm_field(vlm_form_top.get("email"), source="welcome_form")
                fused_record.contact_phone = normalize_vlm_field(vlm_form_top.get("telephone_mobile"), source="welcome_form")

                # Fallback guest name from Form-C if bio scan didn't capture surname or given names
                form_c_name = normalize_vlm_field(vlm_form_top.get("name_of_foreigner"), source="welcome_form")
                if form_c_name.value and (not fused_record.given_names.value and not fused_record.surname.value):
                    parts = form_c_name.value.split(maxsplit=1)
                    fused_record.surname = FieldResult(value=parts[0], confidence=form_c_name.confidence, source="welcome_form")
                    if len(parts) > 1:
                        fused_record.given_names = FieldResult(value=parts[1], confidence=form_c_name.confidence, source="welcome_form")

                # Fallback bio comparisons if passport/visa didn't capture them
                if not fused_record.visa_place_of_issue.value:
                    fused_record.visa_place_of_issue = normalize_vlm_field(vlm_form_top.get("place_of_issue_of_visa"), source="welcome_form")
                if not fused_record.arrival_date_india.value:
                    fused_record.arrival_date_india = normalize_vlm_field(vlm_form_top.get("date_of_arrival_in_india"), source="welcome_form")
                if not fused_record.nationality.value:
                    fused_record.nationality = normalize_vlm_field(vlm_form_top.get("nationality"), source="welcome_form")

                processing_notes.append("Form-C top section (identity & residence) extracted.")
                fused_record.processing_notes = list(processing_notes)
                save_intermediate_job_result(job_id, fused_record.model_dump())

                # ------------------------------------------------------------------
                # Pass 4B: Bottom Section (Rows 6 to 12: Homestay Arrival, Travel & Stay Dates)
                # Crop bottom 58% of the document (from 0.42 * img_h to img_h) to cleanly capture rows 6 to 12
                # ------------------------------------------------------------------
                bottom_crop_box = (0, int(img_h * 0.42), img_w, img_h)
                welcome_bottom_img = welcome_raw.crop(bottom_crop_box)
                logger.info(f"[{job_id}] Pass 4B: Cropped bottom section ({img_w}x{welcome_bottom_img.size[1]}).")

                # Run fast local OCR specifically on the bottom crop and provide all detected cues
                ocr_bottom_text = ""
                try:
                    ocr_bottom_raw = await asyncio.to_thread(pytesseract.image_to_string, welcome_bottom_img)
                    ocr_bottom_lines = [line.strip() for line in ocr_bottom_raw.splitlines() if line.strip()]
                    ocr_bottom_text = "\n".join(ocr_bottom_lines)
                except Exception as e:
                    ocr_bottom_text = "No OCR hints available."

                welcome_bottom_prompt = (
                    "You are an administrative office assistant digitizing the bottom section (Rows 6 to 12) of an official Form C (Rule 14) Arrival Report homestay registration card.\n"
                    "Here are raw OCR character tokens detected on this bottom section to guide your reading:\n"
                    f"\"\"\"\n{ocr_bottom_text}\n\"\"\"\n\n"
                    "Examine the numbered table rows on this slip:\n"
                    "- Row 6: Date and Time of Arrival in the home stay (e.g., date and time like '09.10.16 18:00')\n"
                    "- Row 7: Arrived from (e.g., state or city like 'KERALA')\n"
                    "- Row 8: Proceeding to (from here)\n"
                    "- Row 9: Proposed duration of stay in India\n"
                    "- Row 10: Purpose of visit & your Profession\n"
                    "- Row 11: Expected date of departure (from here)\n"
                    "- Row 12: Country & city you arrived to India\n\n"
                    "Transcribe the genuine handwriting written into each row into this JSON schema:\n"
                    "{\n"
                    '  "datetime_arrival_homestay": {"value": string or null, "confidence": float},\n'
                    '  "arrived_from": {"value": string or null, "confidence": float},\n'
                    '  "proceeding_to": {"value": string or null, "confidence": float},\n'
                    '  "proposed_duration_stay_india": {"value": string or null, "confidence": float},\n'
                    '  "purpose_of_visit_profession": {"value": string or null, "confidence": float},\n'
                    '  "expected_date_departure": {"value": "YYYY-MM-DD" or null, "confidence": float},\n'
                    '  "country_city_arrived_to_india": {"value": string or null, "confidence": float}\n'
                    "}\n"
                    "Instructions:\n"
                    "- Transcribe only genuine handwriting written on these rows.\n"
                    "- If a row is blank or empty, return null for value and 0.0 for confidence.\n"
                    "- Output JSON only."
                )

                vlm_form_bottom = await asyncio.to_thread(
                    run_vlm_inference,
                    welcome_bottom_img,
                    welcome_bottom_prompt,
                    ml_state,
                    "Welcome Form (Bottom Section) VLM",
                    384,
                    65536,
                    262144
                )

                fused_record.arrival_datetime_homestay = normalize_vlm_field(vlm_form_bottom.get("datetime_arrival_homestay"), source="welcome_form")
                fused_record.arrived_from = normalize_vlm_field(vlm_form_bottom.get("arrived_from"), source="welcome_form")
                fused_record.proceeding_to = normalize_vlm_field(vlm_form_bottom.get("proceeding_to"), source="welcome_form")
                fused_record.stay_duration_india = normalize_vlm_field(
                    vlm_form_bottom.get("proposed_duration_stay_india") or vlm_form_bottom.get("stay_duration_india"),
                    source="welcome_form"
                )
                fused_record.purpose_of_visit = normalize_vlm_field(vlm_form_bottom.get("purpose_of_visit_profession"), source="welcome_form")
                fused_record.expected_departure_date = normalize_vlm_field(vlm_form_bottom.get("expected_date_departure"), source="welcome_form")
                fused_record.arrived_country_city_india = normalize_vlm_field(vlm_form_bottom.get("country_city_arrived_to_india"), source="welcome_form")

                processing_notes.append("Form-C bottom section (homestay & travel dates) extracted.")
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
    Safely unwraps both UUID string inputs and dictionary records from startup recovery.

    :param ml_state: Global dictionary containing warm Qwen2-VL model handles.
    """
    logger.info("[*] Background async queue worker loop initialized.")
    while True:
        try:
            job_item = await job_queue.get()
            
            # Defensive unwrap: Extract job_id whether item is a string or dictionary
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