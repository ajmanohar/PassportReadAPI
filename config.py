"""
Configuration management for the Passport Stamp, Visa, Bio-Data & MRZ Capture API.
Loads runtime environment variables dynamically so that host, port, routes,
Tesseract binary paths, and vision-language model parameters are configured cleanly.
"""

import os
import platform
from pydantic_settings import BaseSettings, SettingsConfigDict


def resolve_default_tesseract_binary() -> str:
    """
    Detects the operating system platform and resolves the default path
    to the Tesseract OCR executable.
    """
    system_name = platform.system()
    if system_name == "Windows":
        # Standard Windows installation directory
        return r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    else:
        # Standard Ubuntu / Debian Linux binary location
        return "/usr/bin/tesseract"


def resolve_default_tessdata_dir() -> str:
    """
    Detects the operating system platform and resolves the default directory
    housing language training data (tessdata).
    """
    system_name = platform.system()
    if system_name == "Windows":
        # Standard Windows tessdata directory
        return r"C:\Program Files\Tesseract-OCR\tessdata"
    else:
        # Standard Ubuntu / Debian Linux tessdata location
        return "/usr/share/tesseract-ocr/4.00/tessdata"


class Settings(BaseSettings):
    """
    Central settings model for network binding, route endpoints,
    Tesseract OCR paths, and Vision-Language Model (VLM) execution parameters.
    """
    # --------------------------------------------------------------------------
    # Network Binding Configuration
    # --------------------------------------------------------------------------
    # Bind to 0.0.0.0 to accept traffic forwarded by local tunnels or reverse proxies
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # --------------------------------------------------------------------------
    # API Route Endpoints & Prefixes
    # --------------------------------------------------------------------------
    api_v1_prefix: str = "/api/v1"
    stamp_extract_path: str = "/extract/stamp"
    visa_extract_path: str = "/extract/visa"
    passport_bio_extract_path: str = "/extract/passport-bio"
    mrz_router_prefix: str = "/mrz"

    # --------------------------------------------------------------------------
    # Tesseract OCR & PassportEye Configuration
    # --------------------------------------------------------------------------
    # Resolves binary path with runtime environment variable override support
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", resolve_default_tesseract_binary())
    tessdata_prefix: str = os.getenv("TESSDATA_PREFIX", resolve_default_tessdata_dir())

    # --------------------------------------------------------------------------
    # Vision Language Model (VLM) Architecture & Hardware Allocation
    # --------------------------------------------------------------------------
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    device: str = "cpu"

    # Number of CPU threads assigned to PyTorch compute operations (0 = all cores)
    cpu_threads: int = 0

    # --------------------------------------------------------------------------
    # Vision Token Resolution Bounds (CPU Latency Tuning)
    # --------------------------------------------------------------------------
    max_image_dimension: int = 512
    min_pixels: int = 49 * 28 * 28
    max_pixels: int = 144 * 28 * 28
    max_new_tokens: int = 128

    # --------------------------------------------------------------------------
    # Pydantic Settings Source Settings
    # --------------------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Global singleton instance loaded once across the application lifecycle
settings = Settings()