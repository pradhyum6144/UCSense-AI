"""
UCSense-AI Configuration Module

Centralized configuration management using Pydantic Settings for type-safe
environment variable handling and validation.
"""

from functools import lru_cache
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Application
    app_name: str = "UCSense-AI"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    
    # CORS - use comma-separated string instead of list
    allowed_origins_str: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        alias="ALLOWED_ORIGINS"
    )
    
    @property
    def allowed_origins(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins_str.split(",") if o.strip()]
    
    # AWS Configuration
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_bucket_name: str = "ucsense-uploads"
    
    # Supabase Configuration
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    
    # Processing Configuration
    max_image_size_mb: int = 50
    max_concurrent_jobs: int = 10
    job_timeout_seconds: int = 300
    
    # Model Configuration
    unet_model_path: str = "models/unet_curve_segmentation.h5"
    use_gpu: bool = False
    
    # OCR Configuration
    tesseract_cmd: Optional[str] = None  # Path to tesseract executable
    tesseract_lang: str = "eng"
    
    # Feature Flags
    enable_unet_fallback: bool = True
    enable_confidence_scoring: bool = True
    
    # Thresholds
    image_quality_threshold: float = 0.7  # Below this, use U-Net
    ocr_confidence_threshold: float = 0.6
    curve_smoothness_window: int = 11  # Savitzky-Golay window size
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience instance
settings = get_settings()
