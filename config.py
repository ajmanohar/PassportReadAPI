"""
Central configuration settings for the PassportReadAPI service.
Loads runtime parameters from environment variables with sensible production defaults.
"""

import os
import shutil
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings model defining machine learning runtime thresholds,
    API metadata, hardware execution configurations, and OCR binary paths.
    """
    # API Application Metadata
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

    # Hugging Face Vision-Language Model identifier
    model_id: str = Field(
        default="Qwen/Qwen2-VL-2B-Instruct",
        description="Hugging Face repo or local path for Qwen2-VL vision weights"
    )

    # Execution hardware device target ('cpu', 'cuda', or 'mps')
    device: str = Field(
        default="cpu",
        description="Target compute backend for PyTorch tensor execution"
    )

    # PyTorch CPU compute thread pinning (0 uses system default)
    torch_cpu_threads: int = Field(
        default=4,
        description="Number of CPU threads pinned to PyTorch operations to prevent thermal throttling"
    )

    # Vision processor input token bounding constraints
    min_pixels: int = Field(
        default=256 * 28 * 28,
        description="Minimum visual token resolution boundary for image patching"
    )
    max_pixels: int = Field(
        default=1280 * 28 * 28,
        description="Maximum visual token resolution boundary for memory control"
    )

    # Generation decoding budget
    max_new_tokens: int = Field(
        default=512,
        description="Maximum generated tokens allowed per extraction invocation"
    )

    # OCR and External Binary Execution Paths
    tesseract_cmd: str = Field(
        default="/usr/bin/tesseract",
        description="Absolute path to the system tesseract binary for MRZ parsing"
    )
    system_path_additions: str = Field(
        default="/usr/bin:/bin:/usr/local/bin",
        description="Colon-separated directory paths appended to OS execution PATH"
    )

    class Config:
        # Load environment variables matching case-insensitively
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