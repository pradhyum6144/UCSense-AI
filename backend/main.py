"""
UCSense-AI - Main Application Entry Point

FastAPI application for automated extraction of stress-strain data
from geotechnical UCS (Unconfined Compressive Strength) test graphs.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import settings
from routers import extraction, health

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Pre-warm ML model if enabled
    if settings.enable_unet_fallback:
        try:
            from ml.inference import warm_model
            await warm_model()
            logger.info("U-Net model pre-warmed successfully")
        except Exception as e:
            logger.warning(f"Could not pre-warm U-Net model: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down UCSense-AI")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="AI-powered extraction of stress-strain data from UCS test graphs",
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(extraction.router, prefix="/api/v1", tags=["Extraction"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred"
        }
    )


# Lambda handler for AWS deployment
def handler(event, context):
    """AWS Lambda handler using Mangum adapter."""
    from mangum import Mangum
    asgi_handler = Mangum(app, lifespan="off")
    return asgi_handler(event, context)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
