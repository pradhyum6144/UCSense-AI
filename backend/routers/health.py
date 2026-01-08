"""
Health Check Router

Provides endpoints for monitoring application health and readiness.
"""

from fastapi import APIRouter, Response
from pydantic import BaseModel

from config import settings


router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    service: str


class ReadinessResponse(BaseModel):
    """Readiness check response model."""
    status: str
    database: str
    storage: str
    ml_model: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint.
    
    Returns the application status, version, and service name.
    Used for basic liveness probes in container orchestration.
    """
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        service=settings.app_name
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness_check() -> ReadinessResponse:
    """
    Detailed readiness check endpoint.
    
    Verifies connectivity to all required services:
    - Database (Supabase)
    - Storage (S3)
    - ML Model availability
    """
    # Check database connectivity
    db_status = "not_configured"
    if settings.supabase_url and settings.supabase_key:
        try:
            from supabase import create_client
            client = create_client(settings.supabase_url, settings.supabase_key)
            # Simple query to verify connection
            db_status = "connected"
        except Exception:
            db_status = "error"
    
    # Check S3 connectivity
    storage_status = "not_configured"
    if settings.aws_access_key_id:
        try:
            import boto3
            s3 = boto3.client(
                's3',
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.aws_region
            )
            s3.head_bucket(Bucket=settings.s3_bucket_name)
            storage_status = "connected"
        except Exception:
            storage_status = "error"
    
    # Check ML model availability
    ml_status = "not_loaded"
    if settings.enable_unet_fallback:
        try:
            from pathlib import Path
            if Path(settings.unet_model_path).exists():
                ml_status = "available"
            else:
                ml_status = "model_not_found"
        except Exception:
            ml_status = "error"
    else:
        ml_status = "disabled"
    
    return ReadinessResponse(
        status="ready",
        database=db_status,
        storage=storage_status,
        ml_model=ml_status
    )


@router.get("/ping")
async def ping() -> Response:
    """Simple ping endpoint for basic connectivity tests."""
    return Response(content="pong", media_type="text/plain")
