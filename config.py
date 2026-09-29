"""
Central configuration settings for the PassportReadAPI service.
Loads runtime parameters from environment variables with sensible production defaults.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """
    Application settings model defining machine learning runtime thresholds,
    API metadata, and hardware execution configurations.
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

    class Config:
        # Load environment variables matching case-insensitively
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Global singleton instance consumed across application modules
settings = Settings()