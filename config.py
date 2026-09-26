"""
config.py
---------
Configuration management for the Passport Stamp Capture API.
Loads runtime environment variables dynamically so that host, port,
routes, and model IDs can be changed easily between local development and Ubuntu hosting.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings model.
    Pydantic automatically reads values from the .env file or host environment variables.
    """
    # Network binding settings
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    # API Route paths
    api_v1_prefix: str = "/api/v1"
    stamp_extract_path: str = "/extract/stamp"

    # Vision Language Model parameters
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    device: str = "cpu"
    max_image_dimension: int = 1280

    # Configuration for loading .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Global settings singleton instance initialized at runtime
settings = Settings()