# ==============================================================================
# config.py - PassportReadAPI
# ==============================================================================
# Central configuration settings for the PassportReadAPI service.
# Loads runtime parameters and security credentials from environment variables 
# or local .env file with sensible production defaults.
# Configured specifically for low-latency multithreaded CPU inference on Intel edge hardware.
# ==============================================================================

import os
import shutil
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings model defining machine learning runtime thresholds,
    API metadata, hardware execution configurations, security credentials, and OCR binary paths.
    """
    # -------------------------------------------------------------------------
    # API Application Metadata
    # -------------------------------------------------------------------------
    api_title: str = Field(
        default="Passport and Visa OCR Extraction API",
        description="Public display name for OpenAPI docs and health diagnostic reports"
    )
    api_version: str = Field(
        default="1.0.0",
        description="Semantic application release version"
    )
    api_description: str = Field(
        default=(
            "Dual-engine document extraction combining Qwen2-VL vision models, "
            "Tesseract MRZ check-digit parsing, and asynchronous batch processing."
        ),
        description="Extended documentation narrative"
    )

    # -------------------------------------------------------------------------
    # Security & API Authentication (Pre-Shared Bearer Token)
    # -------------------------------------------------------------------------
    # High-entropy pre-shared secret token for securing endpoints over the public internet.
    # Set via API_BEARER_TOKEN in .env. If left None or empty, auth validation is bypassed.
    api_bearer_token: Optional[str] = Field(
        default=None,
        description="Pre-shared static Bearer token used to authenticate incoming API requests"
    )

    # -------------------------------------------------------------------------
    # Hugging Face Vision-Language Model Identifier
    # -------------------------------------------------------------------------
    model_id: str = Field(
        default="Qwen/Qwen2-VL-2B-Instruct",
        description="Hugging Face repo or local path for Qwen2-VL vision weights"
    )

    # -------------------------------------------------------------------------
    # Execution Hardware Device Target ('cpu', 'cuda', or 'mps')
    # -------------------------------------------------------------------------
    device: str = Field(
        default="cpu",
        description="Target compute backend for PyTorch tensor execution"
    )

    # -------------------------------------------------------------------------
    # PyTorch CPU Compute Thread Pinning
    # -------------------------------------------------------------------------
    torch_cpu_threads: int = Field(
        default=4,
        description="Number of CPU threads pinned to PyTorch operations to prevent thermal throttling"
    )

    # -------------------------------------------------------------------------
    # Image Rescaling & Visual Token Bounding Constraints (CPU Latency Tuning)
    # -------------------------------------------------------------------------
    # Caps the longest image edge before patching to keep attention matrix lightweight
    max_image_dimension: int = Field(
        default=512,
        description="Maximum image width or height in pixels prior to vision model tokenization"
    )

    # Vision processor input token bounding constraints:
    # 49 patches = 49 * 28 * 28 = 38,416 pixels (minimum resolution)
    # 144 patches = 144 * 28 * 28 = 112,896 pixels (maximum resolution)
    # Bounding visual tokens between 49 and 144 reduces CPU token attention time from ~5m to ~15-25s.
    min_pixels: int = Field(
        default=49 * 28 * 28,
        description="Minimum visual token resolution boundary for image patching"
    )
    max_pixels: int = Field(
        default=144 * 28 * 28,
        description="Maximum visual token resolution boundary for CPU memory and attention control"
    )

    # -------------------------------------------------------------------------
    # Generation Decoding Budget
    # -------------------------------------------------------------------------
    max_new_tokens: int = Field(
        default=256,
        description="Maximum generated tokens allowed per extraction invocation"
    )

    # -------------------------------------------------------------------------
    # OCR and External Binary Execution Paths
    # -------------------------------------------------------------------------
    tesseract_cmd: str = Field(
        default="/usr/bin/tesseract",
        description="Absolute path to the system tesseract binary for MRZ parsing"
    )
    system_path_additions: str = Field(
        default="/usr/bin:/bin:/usr/local/bin",
        description="Colon-separated directory paths appended to OS execution PATH"
    )

    class Config:
        # Load environment variables matching case-insensitively from .env
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global singleton instance consumed across application modules
settings = Settings()

# Ensure system binary paths are present in runtime os.environ["PATH"]
# to prevent subprocess lookup failures when executed under minimal systemd environments
current_path = os.environ.get("PATH", "")
path_parts = current_path.split(":") if current_path else []
for directory in settings.system_path_additions.split(":"):
    if directory and directory not in path_parts:
        path_parts.append(directory)
os.environ["PATH"] = ":".join(path_parts)

# Explicitly configure pytesseract binary target if pytesseract is installed in the environment
try:
    import pytesseract
    # Check if configured binary exists; fallback to auto-discovery via which
    resolved_binary = settings.tesseract_cmd if os.path.exists(settings.tesseract_cmd) else shutil.which("tesseract")
    if resolved_binary:
        pytesseract.pytesseract.tesseract_cmd = resolved_binary
except ImportError:
    pass