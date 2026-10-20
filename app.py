# ==============================================================================
# app.py - PassportReadAPI
# ==============================================================================
# FastAPI application for Form-C document information extraction.
# Integrates synchronous Qwen2-VL vision endpoints with per-field confidence scoring,
# Tesseract MRZ parsing, and an asynchronous sequential batch processing queue for multi-document workflows.
# Secured via pre-shared Bearer token authentication on all administrative and job processing routes.
# ==============================================================================

import asyncio
from contextlib import asynccontextmanager
import io
import json
import logging
import os
import re
import secrets
from typing import Dict, Any, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Configuration and modular routers
from config import settings
from routers.mrz_router import router as mrz_router
from routers.jobs_router import router as jobs_router

# Database persistence and async background queue worker
from storage.db import initialize_database, get_all_pending_jobs
from services.job_worker import background_queue_worker, job_queue

# Standardized schema models
from schemas.job import FieldResult
from schemas.passport_bio import PassportBioResponse
from schemas.visa import VisaResponse
from schemas.stamp import StampResponse

# Configure application-level logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("passport_api")


# ==============================================================================
# Security & Authentication Dependency (Pre-Shared Bearer Token)
# ==============================================================================

# Initialize HTTPBearer security scheme with auto_error=False to allow custom error handling
security_scheme = HTTPBearer(auto_error=False)


async def verify_bearer_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_scheme)
) -> None:
    """
    FastAPI dependency enforcing pre-shared Bearer token authentication.
    Compares the incoming authorization token against settings.api_bearer_token
    using constant-time string comparison to prevent timing side-channel attacks.

    If settings.api_bearer_token is None or blank, authentication is bypassed.
    If a token is configured but the request lacks or provides an incorrect token,
    raises HTTP 401 Unauthorized.
    """
    configured_token = settings.api_bearer_token
    if not configured_token:
        # Authentication bypassed if no token configured in environment
        return

    if not credentials or not credentials.credentials:
        logger.warning("[Security] Missing Authorization Bearer header on protected route.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Constant-time comparison prevents timing attacks
    is_valid = secrets.compare_digest(credentials.credentials.strip(), configured_token.strip())
    if not is_valid:
        logger.warning("[Security] Rejected invalid Bearer token attempt.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or unauthorized Bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ==============================================================================
# Global Machine Learning State & Lifespan Management
# ==============================================================================

ml_models: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Initializes Qwen2-VL model and processor into CPU memory on startup,
    launches the asynchronous background queue worker, and cleans up resources on shutdown.
    """
    logger.info(f"[*] Initializing model '{settings.model_id}' on device: {settings.device}...")
    torch.set_num_threads(settings.torch_cpu_threads)
    logger.info(f"[*] PyTorch CPU compute threads configured to: {settings.torch_cpu_threads}")

    try:
        # Load Hugging Face processor and model
        processor = AutoProcessor.from_pretrained(
            settings.model_id,
            min_pixels=settings.min_pixels,
            max_pixels=settings.max_pixels
        )
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            settings.model_id,
            torch_dtype=torch.float32,
            device_map=settings.device,
            low_cpu_mem_usage=True
        )
        model.eval()

        ml_models["model"] = model
        ml_models["processor"] = processor
        logger.info("[*] Model and processor successfully loaded and ready for queries.")
    except Exception as exc:
        logger.error(f"[!] Failed to load model weights: {str(exc)}", exc_info=True)
        raise exc

    # Initialize SQLite database schema
    initialize_database()

    # Recover any queued or in-flight jobs from SQLite and populate in-memory queue
    pending_jobs = get_all_pending_jobs()
    if pending_jobs:
        logger.info(f"[*] Recovered {len(pending_jobs)} pending job(s) from SQLite on startup.")
        for p_job in pending_jobs:
            await job_queue.put(p_job)

    # Launch background queue worker task
    worker_task = asyncio.create_task(background_queue_worker(ml_models))
    logger.info("[*] Background queue worker task started.")

    yield

    # Clean shutdown
    logger.info("[*] Cancelling background queue worker task...")
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass
    ml_models.clear()
    logger.info("[*] Application shutdown complete.")


# Initialize FastAPI application instance
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Public Health Diagnostic Endpoint (Unauthenticated for Tunnel & Uptime Checks)
# ==============================================================================

@app.get("/health", tags=["System Diagnostics"])
async def health_check():
    """
    Public health and diagnostic endpoint.
    Verifies that the API service is active, reports PyTorch device allocation,
    and returns current worker queue size. Left unauthenticated for Cloudflare health checks.
    """
    model_loaded = "model" in ml_models and ml_models["model"] is not None
    processor_loaded = "processor" in ml_models and ml_models["processor"] is not None

    return {
        "status": "healthy" if (model_loaded and processor_loaded) else "degraded",
        "model_loaded": model_loaded,
        "processor_loaded": processor_loaded,
        "device": settings.device,
        "torch_cpu_threads": settings.torch_cpu_threads,
        "queue_size": job_queue.qsize(),
        "auth_enforced": bool(settings.api_bearer_token)
    }


# ==============================================================================
# Router Ingestion with Authentication Dependencies
# ==============================================================================

# Protect jobs router with Bearer token authentication
app.include_router(
    jobs_router,
    prefix="/api/v1",
    tags=["Asynchronous Form-C Batch Pipeline"],
    dependencies=[Depends(verify_bearer_token)]
)

# Protect MRZ router with Bearer token authentication
app.include_router(
    mrz_router,
    prefix="/api/v1/mrz",
    tags=["MRZ Check-Digit Parsing"],
    dependencies=[Depends(verify_bearer_token)]
)