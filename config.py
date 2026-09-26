"""
config.py
---------
Configuration management for the Passport Stamp & Visa Capture API.
Loads runtime environment variables dynamically so that host, port,
routes, and model IDs can be changed easily between local development and Ubuntu hosting.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central settings model for network binding, route endpoints, and VLM parameters.
    """
    # Network binding settings
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # API Route paths
    api_v1_prefix: str = "/api/v1"
    stamp_extract_path: str = "/extract/stamp"
    visa_extract_path: str = "/extract/visa"
    passport_bio_extract_path: str = "/extract/passport-bio"

    # Vision Language Model parameters
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    device: str = "cpu"
    max_image_dimension: int = 1280

    # Load from .env file if present
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Global singleton instance loaded once across the entire application runtime
settings = Settings()