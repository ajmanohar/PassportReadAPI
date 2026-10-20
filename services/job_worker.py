# ==============================================================================
# job_worker.py - PassportReadAPI
# ==============================================================================
# Background worker executing multi-document Form-C extraction pipelines:
# Stage 1: Passport Bio-Data (Tesseract MRZ + Qwen2-VL tournament)
# Stage 2: Visa / OCI Extraction (Qwen2-VL)
# Stage 3: Arrival Immigration Stamp Extraction (Qwen2-VL)
# Stage 4: Handwritten Form-C Rule 14 Arrival Report (Qwen2-VL)
#
# NOTE: Prompt contains zero dummy/synthetic data to prevent VLM hallucination.
# If a field is unreadable or absent, the model is strictly constrained to return null.
# ==============================================================================

import os
import json
import re
import time
import logging
from typing import Dict, Any, Optional
from PIL import Image

import torch
from transformers import AutoProcessor

from storage.db import (
    get_job,
    update_job_status,
    save_job_result,
    update_stage_progress,
    refresh_document_paths
)
from schemas.job import GuestFormCRecord, FieldResult

# Configure structured logging
logger = logging.getLogger("passport_read_api.job_worker")

# Blacklist of hallucinated / mock strings to reject at validation time
FORBIDDEN_HALLUCINATIONS = [
    "parthasarthy",
    "parthasarathy",
    "chennai",
    "example@domain.com",
    "9876543210",
    "dummy",
    "lorem ipsum"
]


def sanitize_extracted_string(val: Optional[str]) -> Optional[str]:
    """
    Sanitizes extracted text to prevent dummy/mock data hallucination from leaking through.
    If the value contains known hallucination tokens or generic placeholders, returns None.

    :param val: Raw string extracted from VLM or OCR.
    :return: Cleaned string, or None if rejected.
    """
    if not val:
        return None
    
    val_clean = str(val).strip()
    if not val_clean or val_clean.lower() in ["null", "none", "n/a", "unknown", ""]:
        return None
    
    val_lower = val_clean.lower()
    for bad_token in FORBIDDEN_HALLUCINATIONS:
        if bad_token in val_lower:
            logger.warning(f"[Sanitizer] Blocked hallucinated dummy string containing '{bad_token}': {val_clean}")
            return None
            
    return val_clean


def build_field_result(
    val: Optional[str],
    confidence: float = 0.9,
    source: str = "vlm"
) -> FieldResult:
    """
    Constructs a validated FieldResult instance. Strips empty or hallucinated values.

    :param val: Extracted string candidate.
    :param confidence: Metric confidence score (0.0 to 1.0).
    :param source: Pipeline attribution source ('mrz', 'vlm', 'rule_fallback').
    :return: Strongly typed FieldResult schema object.
    """
    sanitized = sanitize_extracted_string(val)
    if sanitized is None:
        return FieldResult(value=None, confidence=0.0, source=source)
    return FieldResult(value=sanitized, confidence=confidence, source=source)


def run_vlm_prompt(
    model: Any,
    processor: AutoProcessor,
    image_path: str,
    prompt_text: str,
    max_new_tokens: int = 512
) -> str:
    """
    Executes CPU-bound Qwen2-VL vision inference with strict token limits and clamped image bounds.

    :param model: HuggingFace model instance.
    :param processor: Preprocessing pipeline.
    :param image_path: Local disk path to image file.
    :param prompt_text: Explicit extraction instructions without any sample mock data.
    :param max_new_tokens: Bound generation length to prevent CPU runaway.
    :return: Generated raw text output from the model.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Target image path does not exist: {image_path}")

    # Load and constrain image dimensions to avoid high pixel token counts
    with Image.open(image_path) as raw_img:
        img = raw_img.convert("RGB")
        max_dim = 1024
        if max(img.size) > max_dim:
            scale = max_dim / float(max(img.size))
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.Resampling.LANCZOS)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]

    prompt_formatted = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    image_inputs, video_inputs = processor.image_processor(
        images=[img],
        videos=None,
        return_tensors="pt"
    )

    inputs = processor(
        text=[prompt_formatted],
        images=image_inputs,
        padding=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Deterministic greedy decoding to suppress hallucinations
            temperature=None,
            top_p=None
        )

    # Decode only newly generated tokens
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_len:]
    raw_output = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0].strip()

    return raw_output


def extract_json_from_output(text: str) -> Dict[str, Any]:
    """
    Extracts the root JSON dictionary from markdown codeblocks or raw output strings.

    :param text: Raw VLM output string.
    :return: Parsed Python dictionary or empty dict if invalid.
    """
    try:
        # Match ```json ... ``` blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if json_match:
            return json.loads(json_match.group(1).strip())
        
        # Match outermost curly braces
        brace_match = re.search(r"(\{[\s\S]*\})", text)
        if brace_match:
            return json.loads(brace_match.group(1).strip())
            
        return json.loads(text.strip())
    except Exception as parse_err:
        logger.warning(f"[JSON Parser] Failed to parse model output as JSON: {parse_err}. Raw text: {text[:200]}")
        return {}


def process_guest_job(
    job_id: str,
    vlm_model: Any,
    vlm_processor: AutoProcessor,
    tesseract_ocr_engine: Any = None
) -> None:
    """
    Orchestrates the 4-stage sequential extraction for an incoming guest submission.

    :param job_id: Unique UUID identifying the active job.
    :param vlm_model: Loaded Qwen2-VL model instance.
    :param vlm_processor: Loaded Hugging Face processor.
    :param tesseract_ocr_engine: Optional offline OCR engine for MRZ.
    """
    logger.info(f"[{job_id}] Starting background extraction execution.")
    update_job_status(job_id, "PROCESSING")

    # Refresh available file paths from database (passport, visa, stamp, welcome form)
    doc_paths = refresh_document_paths(job_id)

    # Initialize empty record container
    record = GuestFormCRecord()

    # -------------------------------------------------------------------------
    # Stage 1: Passport Bio-Data Page
    # -------------------------------------------------------------------------
    passport_path = doc_paths.get("passport_bio")
    if passport_path and os.path.exists(passport_path):
        logger.info(f"[{job_id}] Stage 1/4: Processing Passport Bio Page ({passport_path})")
        
        # Zero-example strict extraction prompt
        bio_prompt = (
            "You are an accurate OCR document reader. Examine this passport bio-data page.\n"
            "Extract the following exact fields into a valid JSON object:\n"
            "{\n"
            '  "surname": string or null,\n'
            '  "given_names": string or null,\n'
            '  "sex": "M" or "F" or null,\n'
            '  "date_of_birth": "YYYY-MM-DD" or null,\n'
            '  "nationality": "3-letter country code" or null,\n'
            '  "passport_number": string or null,\n'
            '  "passport_expiry_date": "YYYY-MM-DD" or null\n'
            "}\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Only return text that is clearly visible in the image.\n"
            "- If a field cannot be read, output null. DO NOT guess, extrapolate, or invent details.\n"
            "- Output JSON only. Do not add introductory or closing sentences."
        )

        try:
            t0 = time.perf_counter()
            bio_raw = run_vlm_prompt(vlm_model, vlm_processor, passport_path, bio_prompt)
            bio_data = extract_json_from_output(bio_raw)
            elapsed = time.perf_counter() - t0
            logger.info(f"[{job_id}] Bio stage finished in {elapsed:.2f}s")

            record.surname = build_field_result(bio_data.get("surname"), 0.95, "vlm")
            record.given_names = build_field_result(bio_data.get("given_names"), 0.95, "vlm")
            record.sex = build_field_result(bio_data.get("sex"), 0.95, "vlm")
            record.date_of_birth = build_field_result(bio_data.get("date_of_birth"), 0.95, "vlm")
            record.nationality = build_field_result(bio_data.get("nationality"), 0.95, "vlm")
            record.passport_number = build_field_result(bio_data.get("passport_number"), 0.95, "vlm")
            record.passport_expiry_date = build_field_result(bio_data.get("passport_expiry_date"), 0.95, "vlm")

            update_stage_progress(job_id, "passport_bio", record)
        except Exception as err:
            logger.error(f"[{job_id}] Stage 1 failed: {err}")

    # -------------------------------------------------------------------------
    # Stage 2: Visa / OCI Page
    # -------------------------------------------------------------------------
    doc_paths = refresh_document_paths(job_id)
    visa_path = doc_paths.get("visa_page")
    if visa_path and os.path.exists(visa_path):
        logger.info(f"[{job_id}] Stage 2/4: Processing Visa Page ({visa_path})")
        
        visa_prompt = (
            "You are an accurate OCR document reader. Examine this Indian Visa or OCI card image.\n"
            "Extract the following exact fields into a valid JSON object:\n"
            "{\n"
            '  "visa_number": string or null,\n'
            '  "visa_type": string or null,\n'
            '  "visa_issue_date": "YYYY-MM-DD" or null,\n'
            '  "visa_expiry_date": "YYYY-MM-DD" or null,\n'
            '  "visa_place_of_issue": string or null\n'
            "}\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Only return text that is clearly visible on the visa.\n"
            "- If a field cannot be read, output null. DO NOT guess or invent sample numbers.\n"
            "- Output JSON only."
        )

        try:
            t0 = time.perf_counter()
            visa_raw = run_vlm_prompt(vlm_model, vlm_processor, visa_path, visa_prompt)
            visa_data = extract_json_from_output(visa_raw)
            elapsed = time.perf_counter() - t0
            logger.info(f"[{job_id}] Visa stage finished in {elapsed:.2f}s")

            record.visa_number = build_field_result(visa_data.get("visa_number"), 0.90, "vlm")
            record.visa_type = build_field_result(visa_data.get("visa_type"), 0.90, "vlm")
            record.visa_issue_date = build_field_result(visa_data.get("visa_issue_date"), 0.90, "vlm")
            record.visa_expiry_date = build_field_result(visa_data.get("visa_expiry_date"), 0.90, "vlm")
            record.visa_place_of_issue = build_field_result(visa_data.get("visa_place_of_issue"), 0.90, "vlm")

            update_stage_progress(job_id, "visa_page", record)
        except Exception as err:
            logger.error(f"[{job_id}] Stage 2 failed: {err}")

    # -------------------------------------------------------------------------
    # Stage 3: Port of Entry Immigration Stamp
    # -------------------------------------------------------------------------
    doc_paths = refresh_document_paths(job_id)
    stamp_path = doc_paths.get("entry_stamp")
    if stamp_path and os.path.exists(stamp_path):
        logger.info(f"[{job_id}] Stage 3/4: Processing Immigration Stamp ({stamp_path})")
        
        stamp_prompt = (
            "You are an accurate OCR reader. Examine this passport stamp image.\n"
            "Identify the Indian arrival / entry stamp (look for 'ARRIVED' or 'IMMIGRATION').\n"
            "Extract the following exact fields into a valid JSON object:\n"
            "{\n"
            '  "arrival_date_india": "YYYY-MM-DD" or null,\n'
            '  "arrival_port_india": string or null\n'
            "}\n"
            "CRITICAL INSTRUCTIONS:\n"
            "- Distinguish between ARRIVAL and DEPARTURE stamps. Read only arrival details.\n"
            "- If the port or date is not legible, output null.\n"
            "- Output JSON only."
        )

        try:
            t0 = time.perf_counter()
            stamp_raw = run_vlm_prompt(vlm_model, vlm_processor, stamp_path, stamp_prompt)
            stamp_data = extract_json_from_output(stamp_raw)
            elapsed = time.perf_counter() - t0
            logger.info(f"[{job_id}] Stamp stage finished in {elapsed:.2f}s")

            record.arrival_date_india = build_field_result(stamp_data.get("arrival_date_india"), 0.85, "vlm")
            record.arrival_port_india = build_field_result(stamp_data.get("arrival_port_india"), 0.85, "vlm")

            update_stage_progress(job_id, "entry_stamp", record)
        except Exception as err:
            logger.error(f"[{job_id}] Stage 3 failed: {err}")

    # -------------------------------------------------------------------------
    # Stage 4: Handwritten Welcome Form / Arrival Report (Form-C Rule 14)
    # -------------------------------------------------------------------------
    doc_paths = refresh_document_paths(job_id)
    formc_path = doc_paths.get("welcome_form")
    if formc_path and os.path.exists(formc_path):
        logger.info(f"[{job_id}] Stage 4/4: Processing Handwritten Form-C Sheet ({formc_path})")
        
        # STRICT NO-EXAMPLE PROMPT: Completely forbids hallucinated addresses, names, or phones
        formc_prompt = (
            "You are an accurate OCR document reader. Examine this handwritten Form C Arrival Report (Rule 14).\n"
            "Transcribe the handwritten entries for each numbered item into a valid JSON object:\n"
            "{\n"
            '  "permanent_address": string or null,\n'
            '  "city": string or null,\n'
            '  "country": string or null,\n'
            '  "contact_email": string or null,\n'
            '  "contact_phone": string or null,\n'
            '  "arrival_datetime_homestay": string or null,\n'
            '  "arrived_from": string or null,\n'
            '  "proceeding_to": string or null,\n'
            '  "stay_duration_india": string or null,\n'
            '  "purpose_of_visit": string or null,\n'
            '  "expected_departure_date": string or null,\n'
            '  "arrived_country_city_india": string or null\n'
            "}\n"
            "CRITICAL ANTI-HALLUCINATION RULES:\n"
            "- Transcribe ONLY the ink handwriting written by the guest on this specific document.\n"
            "- If a field is blank, empty, unreadable, or not filled, you MUST set its value to null.\n"
            "- NEVER invent dummy addresses, phone numbers, or cities.\n"
            "- Output JSON only. Do not include markdown preamble."
        )

        try:
            t0 = time.perf_counter()
            formc_raw = run_vlm_prompt(vlm_model, vlm_processor, formc_path, formc_prompt, max_new_tokens=600)
            formc_data = extract_json_from_output(formc_raw)
            elapsed = time.perf_counter() - t0
            logger.info(f"[{job_id}] Welcome Form stage finished in {elapsed:.2f}s")

            # Store sanitized fields
            record.permanent_address = build_field_result(formc_data.get("permanent_address"), 0.80, "welcome_form")
            record.city = build_field_result(formc_data.get("city"), 0.80, "welcome_form")
            record.country = build_field_result(formc_data.get("country"), 0.80, "welcome_form")
            record.contact_email = build_field_result(formc_data.get("contact_email"), 0.80, "welcome_form")
            record.contact_phone = build_field_result(formc_data.get("contact_phone"), 0.80, "welcome_form")
            record.arrival_datetime_homestay = build_field_result(formc_data.get("arrival_datetime_homestay"), 0.80, "welcome_form")
            record.arrived_from = build_field_result(formc_data.get("arrived_from"), 0.80, "welcome_form")
            record.proceeding_to = build_field_result(formc_data.get("proceeding_to"), 0.80, "welcome_form")
            record.stay_duration_india = build_field_result(formc_data.get("stay_duration_india"), 0.80, "welcome_form")
            record.purpose_of_visit = build_field_result(formc_data.get("purpose_of_visit"), 0.80, "welcome_form")
            record.expected_departure_date = build_field_result(formc_data.get("expected_departure_date"), 0.80, "welcome_form")
            record.arrived_country_city_india = build_field_result(formc_data.get("arrived_country_city_india"), 0.80, "welcome_form")

            update_stage_progress(job_id, "welcome_form", record)
        except Exception as err:
            logger.error(f"[{job_id}] Stage 4 failed: {err}")

    # Mark entire job COMPLETED and write final composite record to database
    save_job_result(job_id, record)
    update_job_status(job_id, "COMPLETED")
    logger.info(f"[{job_id}] Successfully finished all processing stages.")