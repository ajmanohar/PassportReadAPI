"""
Background batch execution pipeline and serial queue worker for asynchronous Form-C jobs.
Processes multi-document bundles (Passport Bio, Visa, Entry Stamp, Welcome Form) sequentially.
Executes an unconditional 'best-of-both' confidence tournament between Tesseract MRZ and
Qwen2-VL for the Passport Bio page, selecting the highest-confidence extraction per field.
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
from schemas.job import GuestFormCRecord, FieldResult

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
    Loads an image from disk storage, standardizes color channels to RGB,
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
# Tournament Selection & Normalization Helpers
# ==============================================================================

def normalize_vlm_field(raw_field: Any, default_score: float = 0.80) -> FieldResult:
    """
    Parses a nested or scalar field returned by Qwen2-VL into a validated FieldResult.

    :param raw_field: Dict containing {'value': ..., 'confidence': ...} or a raw scalar.
    :param default_score: Default confidence score if none is specified by the model.
    :return: Sanitized FieldResult instance.
    """
    if raw_field is None:
        return FieldResult(value=None, confidence=0.0, source="qwen2_vl")

    if isinstance(raw_field, dict):
        val = raw_field.get("value")
        raw_conf = raw_field.get("confidence", default_score)
    else:
        val = str(raw_field)
        raw_conf = default_score

    if not val or str(val).strip().lower() in ["null", "none", ""]:
        return FieldResult(value=None, confidence=0.0, source="qwen2_vl")

    try:
        conf_float = float(raw_conf)
        conf_float = max(0.0, min(1.0, conf_float))
    except (ValueError, TypeError):
        conf_float = default_score

    return FieldResult(
        value=str(val).strip(),
        confidence=round(conf_float, 2),
        source="qwen2_vl"
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
# Job Processing Pipeline
# ==============================================================================

async def process_single_job(job_id: str, ml_state: dict) -> None:
    """
    Executes the multi-document extraction pipeline for a single queued job.
    Runs Tesseract MRZ and Qwen2-VL in parallel for the bio page, selects the
    best candidate per field, and continues through Visa, Stamp, and Welcome Form.

    :param job_id: Unique UUID string identifier of the target job.
    :param ml_state: Shared dictionary holding model handles.
    """
    logger.info(f"[*] Starting processing for job: {job_id}")
    update_job_status(job_id, "PROCESSING")

    job_record = get_job(job_id)
    if not job_record:
        logger.error(f"Job record {job_id} not found in database.")
        return

    doc_paths: Dict[str, str] = job_record.get("doc_paths", {})
    fused_record = GuestFormCRecord()
    processing_notes = []

    try:
        # ======================================================================
        # PASSPORT BIO PAGE: UNCONDITIONAL DUAL-ENGINE TOURNAMENT
        # ======================================================================
        bio_path = doc_paths.get("passport_bio")
        if bio_path and os.path.exists(bio_path):
            logger.info(f"[{job_id}] Initiating dual-engine extraction for passport bio page...")

            # --- Engine A: Tesseract MRZ Execution ---
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

                    processing_notes.append(f"MRZ parsed (valid: {valid_mrz_flag}, score: {valid_score}).")
                else:
                    processing_notes.append("MRZ not detected on passport image.")
            except Exception as mrz_exc:
                logger.warning(f"[{job_id}] MRZ extraction error: {str(mrz_exc)}")
                processing_notes.append(f"MRZ exception: {str(mrz_exc)}")

            # --- Engine B: Qwen2-VL Visual Bio Execution ---
            vlm_candidates: Dict[str, FieldResult] = {}
            bio_image = await asyncio.to_thread(load_and_preprocess_image, bio_path)

            if bio_image:
                logger.info(f"[{job_id}] Running Qwen2-VL visual bio extraction...")
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
                vlm_bio = await asyncio.to_thread(run_vlm_inference, bio_image, bio_prompt, ml_state)

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

        # ======================================================================
        # VISA / OCI EXTRACTION (QWEN2-VL)
        # ======================================================================
        visa_path = doc_paths.get("visa_page")
        if visa_path and os.path.exists(visa_path):
            logger.info(f"[{job_id}] Running Qwen2-VL visa extraction...")
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
                vlm_visa = await asyncio.to_thread(run_vlm_inference, visa_image, visa_prompt, ml_state)
                fused_record.visa_number = normalize_vlm_field(vlm_visa.get("visa_number"))
                fused_record.visa_type = normalize_vlm_field(vlm_visa.get("visa_type"))
                fused_record.visa_issue_date = normalize_vlm_field(vlm_visa.get("issue_date"))
                fused_record.visa_expiry_date = normalize_vlm_field(vlm_visa.get("expiry_date"))
                fused_record.visa_place_of_issue = normalize_vlm_field(vlm_visa.get("place_of_issue"))
                processing_notes.append("Visa / OCI details extracted via Qwen2-VL.")

        # ======================================================================
        # IMMIGRATION ENTRY STAMP EXTRACTION (QWEN2-VL)
        # ======================================================================
        stamp_path = doc_paths.get("entry_stamp")
        if stamp_path and os.path.exists(stamp_path):
            logger.info(f"[{job_id}] Running Qwen2-VL entry stamp extraction...")
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
                vlm_stamp = await asyncio.to_thread(run_vlm_inference, stamp_image, stamp_prompt, ml_state)
                fused_record.arrival_date_india = normalize_vlm_field(vlm_stamp.get("arrival_date"))
                fused_record.arrival_port_india = normalize_vlm_field(vlm_stamp.get("arrival_port"))
                processing_notes.append("Entry stamp details extracted via Qwen2-VL.")

        # ======================================================================
        # HANDWRITTEN WELCOME FORM EXTRACTION (FORM-C RULE 14)
        # ======================================================================
        welcome_path = doc_paths.get("welcome_form")
        if welcome_path and os.path.exists(welcome_path):
            logger.info(f"[{job_id}] Running Qwen2-VL handwritten welcome form extraction...")
            welcome_image = await asyncio.to_thread(load_and_preprocess_image, welcome_path)
            if welcome_image:
                welcome_prompt = (
                    "Extract handwritten Form C Arrival Report fields from the image. "
                    "For each field provide value and confidence (0.0 - 1.0). "
                    "Output JSON formatted exactly as:\n"
                    "{\n"
                    '  "full_address": {"value": "...", "confidence": 0.80},\n'
                    '  "email": {"value": "...", "confidence": 0.80},\n'
                    '  "mobile": {"value": "...", "confidence": 0.80},\n'
                    '  "arrived_from": {"value": "...", "confidence": 0.80},\n'
                    '  "proceeding_to": {"value": "...", "confidence": 0.80},\n'
                    '  "purpose_of_visit_profession": {"value": "...", "confidence": 0.80}\n'
                    "}"
                )
                vlm_form = await asyncio.to_thread(run_vlm_inference, welcome_image, welcome_prompt, ml_state)
                fused_record.permanent_address = normalize_vlm_field(vlm_form.get("full_address"))
                fused_record.contact_email = normalize_vlm_field(vlm_form.get("email"))
                fused_record.contact_phone = normalize_vlm_field(vlm_form.get("mobile"))
                fused_record.arrived_from = normalize_vlm_field(vlm_form.get("arrived_from"))
                fused_record.proceeding_to = normalize_vlm_field(vlm_form.get("proceeding_to"))
                fused_record.purpose_of_visit = normalize_vlm_field(vlm_form.get("purpose_of_visit_profession"))
                processing_notes.append("Handwritten welcome form extracted via Qwen2-VL.")

        # Persist final fused record with diagnostic tournament logs
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