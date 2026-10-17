"""
Background batch execution pipeline and serial queue worker for asynchronous Form-C jobs.
Processes multi-document bundles (Passport Bio, Visa, Entry Stamp, Welcome Form) sequentially.
Executes an unconditional 'best-of-both' confidence tournament between Tesseract MRZ and
Qwen2-VL for the Passport Bio page, selecting the highest-confidence extraction per field.
Persists progress and results directly to the SQLite database with detailed latency telemetry.

Tuned specifically for CPU execution:
- Caps visual tokens via settings.min_pixels and settings.max_pixels (49 - 144 patches).
- Downscales images to settings.max_image_dimension (512px) to prevent attention matrix latency blowouts.
- Enforces strict PyTorch thread allocation to all physical CPU cores.
"""

import asyncio
import io
import json
import logging
import os
import re
import time
from typing import Dict, Any, Optional, List

from PIL import Image
import torch
from passporteye import read_mrz

# Central configuration and repository imports
from config import settings
from storage.db import update_job_status, save_job_result, get_job
from schemas.job import GuestFormCRecord, FieldResult

# Set up module-level logger
logger = logging.getLogger("job_worker")
logger.setLevel(logging.INFO)

# Global in-memory queue holding job UUID strings for sequential execution
job_queue: asyncio.Queue = asyncio.Queue()

# Base disk directory for persisting uploaded document images during asynchronous processing
BASE_STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "storage", "jobs")


# ==============================================================================
# Image Preparation Helper
# ==============================================================================

def load_and_preprocess_image(
    file_path: str,
    max_dimension: Optional[int] = None
) -> Optional[Image.Image]:
    """
    Loads an image from disk storage, standardizes color channels to RGB,
    logs its native geometry, and downscales proportionally if the maximum
    dimension exceeds configured limits (defaulting to settings.max_image_dimension: 512px).
    Bounding the longest edge to 512px bounds visual tokens between 49 and 144 patches,
    keeping CPU generation latency under 15-25 seconds per stage.

    :param file_path: Absolute or relative filesystem path to the image file.
    :param max_dimension: Maximum allowed width or height in pixels. Defaults to settings.max_image_dimension.
    :return: Prepared PIL Image instance, or None if the file cannot be read.
    """
    if not os.path.exists(file_path):
        logger.warning(f"[ImageLoader] File not found on disk: {file_path}")
        return None

    # Resolve target dimension from settings if not explicitly provided
    target_max_dim = max_dimension if max_dimension is not None else settings.max_image_dimension

    try:
        start_t = time.perf_counter()
        image = Image.open(file_path)
        orig_w, orig_h = image.size

        # Standardize color profile to 3-channel RGB (removes alpha or CMYK artifacts)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Downscale proportionally to bound attention matrix computation while preserving sharp text
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

    :param image: Preprocessed PIL Image instance bounded to 512px.
    :param system_prompt: Guidance text instructing the VLM on expected JSON fields.
    :param ml_state: Shared dictionary holding 'model' and 'processor' instances.
    :param stage_label: Diagnostic label for logging (e.g. 'Passport Bio', 'Visa', etc.).
    :return: Extracted key-value dictionary parsed from the model's generated JSON.
    """
    from qwen_vl_utils import process_vision_info

    processor = ml_state.get("processor")
    model = ml_state.get("model")

    if not model or not processor:
        raise RuntimeError("Qwen2-VL model and processor are not loaded in ml_state.")

    # Explicitly pin PyTorch compute threads to prevent thread contention on CPU
    torch.set_num_threads(settings.torch_cpu_threads)

    start_vlm_time = time.perf_counter()
    logger.info(f"[{stage_label}] Preparing prompt tensor inputs (image: {image.size[0]}x{image.size[1]})...")

    # Format user prompt according to Qwen2-VL chat conversation template
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    # Bound visual tokens strictly between 49 and 144 patches for fast CPU inference
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

    logger.info(f"[{stage_label}] Beginning PyTorch CPU token generation (max_new_tokens={settings.max_new_tokens})...")
    gen_start = time.perf_counter()

    # Run deterministic inference without calculating gradients
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
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

    # Extract JSON string payload bounded by braces
    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            parsed_json = json.loads(match.group(0))
            return parsed_json
        return json.loads(raw_text)
    except Exception as parse_err:
        logger.warning(
            f"[{stage_label}] Failed to parse structured JSON from model response: {parse_err}. "
            f"Raw text excerpt: {raw_text[:250]}"
        )
        return {"confidence_note": f"Raw model response: {raw_text}"}


# ==============================================================================
# Tournament Selection & Normalization Helpers
# ==============================================================================

def normalize_vlm_field(raw_field: Any, default_score: float = 0.80, source: str = "qwen2_vl") -> FieldResult:
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

    if not val or str(val).strip().lower() in ["null", "none", "", "n/a", "unknown"]:
        return FieldResult(value=None, confidence=0.0, source=source)

    try:
        conf_float = float(raw_conf)
        conf_float = max(0.0, min(1.0, conf_float))
    except (ValueError, TypeError):
        conf_float = default_score

    return FieldResult(
        value=str(val).strip(),
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
    # If one candidate has no value, automatically choose the other
    if cand_mrz.value and not cand_vlm.value:
        return cand_mrz
    if cand_vlm.value and not cand_mrz.value:
        return cand_vlm
    if not cand_mrz.value and not cand_vlm.value:
        return FieldResult(value=None, confidence=0.0, source="unassigned")

    # Both have values: compare confidence scores
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


# ==============================================================================
# Dynamic Document Discovery Helper
# ==============================================================================

def refresh_document_paths(job_id: str) -> Dict[str, str]:
    """
    Queries SQLite and inspects the on-disk job directory to discover any files
    attached after the initial job ingestion. Guarantees that documents uploaded
    while previous stages were executing are never missed.

    :param job_id: UUID tracking string.
    :return: Fresh mapping of document types to verified local file paths.
    """
    job_record = get_job(job_id)
    doc_paths: Dict[str, str] = job_record.get("doc_paths", {}) if job_record else {}

    job_dir = os.path.join(BASE_STORAGE_DIR, job_id)
    if os.path.isdir(job_dir):
        # Scan filesystem for standard attachment prefix names
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

                    valid_mrz_flag = bool(check_composite or valid_score >= 80)

                    def mk_checked(val: Optional[str], check_passed: bool) -> FieldResult:
                        if not val:
                            return FieldResult(value=None, confidence=0.0, source="tesseract_mrz")
                        return FieldResult(
                            value=val,
                            confidence=0.99 if check_passed else 0.30,
                            source="tesseract_mrz"
                        )

                    def mk_text(val: Optional[str]) -> FieldResult:
                        if not val:
                            return FieldResult(value=None, confidence=0.0, source="tesseract_mrz")
                        return FieldResult(
                            value=val,
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

            # --- Engine B: Qwen2-VL Visual Bio Execution ---
            vlm_candidates: Dict[str, FieldResult] = {}
            # Preprocess image downscaled to settings.max_image_dimension (512px)
            bio_image = await asyncio.to_thread(load_and_preprocess_image, bio_path)

            if bio_image:
                bio_prompt = (
                    "Carefully read the passport bio page image. For each field, provide the extracted string value "
                    "and your visual confidence score between 0.0 and 1.0 based on clarity and legibility. "
                    "Return valid JSON formatted exactly with this structure:\n"
                    "{\n"
                    '  "surname": {"value": "...", "confidence": 0.95},\n'
                    '  "given_names": {"value": "...", "confidence": 0.95},\n'
                    '  "passport_number": {"value": "...", "confidence": 0.90},\n'
                    '  "nationality": {"value": "...", "confidence": 0.95},\n'
                    '  "date_of_birth": {"value": "YYYY-MM-DD", "confidence": 0.90},\n'
                    '  "sex": {"value": "M or F", "confidence": 0.95},\n'
                    '  "expiry_date": {"value": "YYYY-MM-DD", "confidence": 0.90},\n'
                    '  "confidence_note": "short explanation of image quality"\n'
                    "}"
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
            # Downscale image to 512px
            visa_image = await asyncio.to_thread(load_and_preprocess_image, visa_path)
            if visa_image:
                visa_prompt = (
                    "Extract visa details from the image. For each field provide value and confidence (0.0 - 1.0). "
                    "Output JSON formatted exactly as:\n"
                    "{\n"
                    '  "visa_number": {"value": "...", "confidence": 0.90},\n'
                    '  "visa_type": {"value": "...", "confidence": 0.90},\n'
                    '  "issue_date": {"value": "YYYY-MM-DD", "confidence": 0.85},\n'
                    '  "expiry_date": {"value": "YYYY-MM-DD", "confidence": 0.85},\n'
                    '  "place_of_issue": {"value": "...", "confidence": 0.85}\n'
                    "}"
                )
                vlm_visa = await asyncio.to_thread(run_vlm_inference, visa_image, visa_prompt, ml_state, "Visa VLM")
                fused_record.visa_number = normalize_vlm_field(vlm_visa.get("visa_number"))
                fused_record.visa_type = normalize_vlm_field(vlm_visa.get("visa_type"))
                fused_record.visa_issue_date = normalize_vlm_field(vlm_visa.get("issue_date"))
                fused_record.visa_expiry_date = normalize_vlm_field(vlm_visa.get("expiry_date"))
                fused_record.visa_place_of_issue = normalize_vlm_field(vlm_visa.get("place_of_issue"))
                processing_notes.append("Visa / OCI details extracted via Qwen2-VL.")
        else:
            logger.info(f"[{job_id}] Stage 2/4: No visa_page document attached to job. Skipping.")

        # ======================================================================
        # 3. IMMIGRATION ENTRY STAMP EXTRACTION (QWEN2-VL)
        # ======================================================================
        doc_paths = refresh_document_paths(job_id)
        stamp_path = doc_paths.get("entry_stamp")

        if stamp_path and os.path.exists(stamp_path):
            logger.info(f"[{job_id}] Stage 3/4: Running Qwen2-VL entry stamp extraction ({stamp_path})...")
            # Downscale image to 512px
            stamp_image = await asyncio.to_thread(load_and_preprocess_image, stamp_path)
            if stamp_image:
                stamp_prompt = (
                    "Extract arrival stamp details from the image. For each field provide value and confidence (0.0 - 1.0). "
                    "Output JSON formatted exactly as:\n"
                    "{\n"
                    '  "arrival_date": {"value": "YYYY-MM-DD", "confidence": 0.85},\n'
                    '  "arrival_port": {"value": "...", "confidence": 0.85}\n'
                    "}"
                )
                vlm_stamp = await asyncio.to_thread(run_vlm_inference, stamp_image, stamp_prompt, ml_state, "Stamp VLM")
                fused_record.arrival_date_india = normalize_vlm_field(vlm_stamp.get("arrival_date"))
                fused_record.arrival_port_india = normalize_vlm_field(vlm_stamp.get("arrival_port"))
                processing_notes.append("Entry stamp details extracted via Qwen2-VL.")
        else:
            logger.info(f"[{job_id}] Stage 3/4: No entry_stamp document attached to job. Skipping.")

        # ======================================================================
        # 4. HANDWRITTEN ARRIVAL REPORT EXTRACTION (FORM-C RULE 14)
        # ======================================================================
        doc_paths = refresh_document_paths(job_id)
        welcome_path = doc_paths.get("welcome_form")

        if welcome_path and os.path.exists(welcome_path):
            logger.info(f"[{job_id}] Stage 4/4: Running Qwen2-VL handwritten Form-C extraction ({welcome_path})...")
            # Downscale image to 512px
            welcome_image = await asyncio.to_thread(load_and_preprocess_image, welcome_path)
            if welcome_image:
                welcome_prompt = (
                    "Carefully read the handwritten ARRIVAL REPORT (THE REGISTRATION OF FOREIGNERS RULES 1939 FORM C RULE 14) "
                    "document image. Extract every numbered item accurately. "
                    "For each field, return a JSON object with 'value' and 'confidence' (0.0 - 1.0).\n"
                    "Required JSON format:\n"
                    "{\n"
                    '  "name_of_foreigner": {"value": "...", "confidence": 0.85},\n'
                    '  "full_address": {"value": "...", "confidence": 0.80},\n'
                    '  "city": {"value": "...", "confidence": 0.80},\n'
                    '  "country": {"value": "...", "confidence": 0.80},\n'
                    '  "email": {"value": "...", "confidence": 0.85},\n'
                    '  "telephone_mobile": {"value": "...", "confidence": 0.85},\n'
                    '  "nationality": {"value": "...", "confidence": 0.80},\n'
                    '  "place_of_issue_of_visa": {"value": "...", "confidence": 0.80},\n'
                    '  "date_of_arrival_in_india": {"value": "YYYY-MM-DD", "confidence": 0.80},\n'
                    '  "datetime_arrival_homestay": {"value": "...", "confidence": 0.80},\n'
                    '  "arrived_from": {"value": "...", "confidence": 0.80},\n'
                    '  "proceeding_to": {"value": "...", "confidence": 0.80},\n'
                    '  "proposed_duration_stay_india": {"value": "...", "confidence": 0.80},\n'
                    '  "purpose_of_visit_profession": {"value": "...", "confidence": 0.80},\n'
                    '  "expected_date_departure": {"value": "YYYY-MM-DD", "confidence": 0.80},\n'
                    '  "country_city_arrived_to_india": {"value": "...", "confidence": 0.80}\n'
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