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
    # Bind to 0.0.0.0 to accept requests forwarded from local Cloudflare tunnel daemon
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
    # Set to 0 to let PyTorch automatically detect and utilize all available physical CPU cores.
    cpu_threads: int = 0

    # --------------------------------------------------------------------------
    # Vision Token Resolution Bounds (CPU Latency Tuning)
    # --------------------------------------------------------------------------
    # Downscale dimension before passing into processor
    max_image_dimension: int = 512

    # Minimum visual tokens (49 patches * 28 * 28 pixels = 38,416 pixels)
    min_pixels: int = 49 * 28 * 28

    # Maximum visual tokens (144 patches * 28 * 28 pixels = 112,896 pixels)
    # 144 patches drops attention computation dramatically, ensuring CPU
    # processing completes in 15-25 seconds and safely beats Cloudflare's 100s limit.
    max_pixels: int = 144 * 28 * 28

    # Upper bound on generated response tokens (JSON bio response is ~90 tokens)
    max_new_tokens: int = 128

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