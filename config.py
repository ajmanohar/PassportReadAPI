"""
Configuration management for the Passport Stamp, Visa & Bio-Data Capture API.
Loads runtime environment variables dynamically so that host, port,
routes, and vision-language model parameters can be configured easily.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central settings model for network binding, route endpoints,
    and Vision-Language Model (VLM) CPU execution parameters.
    """
    # --------------------------------------------------------------------------
    # Network Binding Configuration
    # --------------------------------------------------------------------------
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # --------------------------------------------------------------------------
    # API Route Endpoints
    # --------------------------------------------------------------------------
    api_v1_prefix: str = "/api/v1"
    stamp_extract_path: str = "/extract/stamp"
    visa_extract_path: str = "/extract/visa"
    passport_bio_extract_path: str = "/extract/passport-bio"

    # --------------------------------------------------------------------------
    # Vision Language Model (VLM) Architecture & Hardware Allocation
    # --------------------------------------------------------------------------
    # Hugging Face repository identifier for Qwen2-VL-2B
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"

    # Target compute device ('cpu' for Intel Dell Micro without Nvidia GPU)
    device: str = "cpu"

    # Number of CPU threads assigned to PyTorch compute operations.
    # Set to 0 to let PyTorch automatically utilize all available physical CPU cores.
    cpu_threads: int = 0

    # --------------------------------------------------------------------------
    # Vision Token Resolution Bounds (CPU Latency Tuning)
    # --------------------------------------------------------------------------
    # Downscale ceiling applied before feeding into the VLM processor
    max_image_dimension: int = 1024

    # Minimum visual tokens (128 patches * 28 * 28 pixels = 100,352 pixels)
    min_pixels: int = 128 * 28 * 28

    # Maximum visual tokens (512 patches * 28 * 28 pixels = 401,408 pixels)
    # Restricting to 512 tokens ensures fast CPU inference (<30s) to prevent
    # Cloudflare Error 524 (100s timeout) while retaining sharp document text.
    max_pixels: int = 512 * 28 * 28

    # Upper bound on generated response tokens (JSON bio response is ~150 tokens)
    max_new_tokens: int = 256

    # --------------------------------------------------------------------------
    # Pydantic Settings Source Settings
    # --------------------------------------------------------------------------
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Instantiate a singleton configuration object for use throughout the application
settings = Settings()