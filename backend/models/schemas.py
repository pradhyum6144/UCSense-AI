"""
Pydantic Schemas for API Request/Response Models

Defines all data models used in the UCSense-AI API.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl
import uuid


class JobStatus(str, Enum):
    """Extraction job status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ExtractionMethod(str, Enum):
    """Curve extraction method."""
    CONTOUR = "contour"
    UNET = "unet"
    HYBRID = "hybrid"
    AUTO = "auto"


class DataPoint(BaseModel):
    """Single data point on the stress-strain curve."""
    strain: float = Field(..., description="Strain value (%)")
    stress: float = Field(..., description="Stress value (kN/m²)")
    pixel_x: Optional[int] = Field(None, description="Original pixel X coordinate")
    pixel_y: Optional[int] = Field(None, description="Original pixel Y coordinate")


class UCSFeatureResponse(BaseModel):
    """Extracted UCS test features."""
    peak_stress: float = Field(..., description="Peak UCS value (kN/m²)")
    failure_strain: float = Field(..., description="Strain at peak stress (%)")
    initial_modulus: Optional[float] = Field(None, description="Initial tangent modulus")
    secant_modulus_50: Optional[float] = Field(None, description="Secant modulus at 50% peak")
    energy_to_peak: Optional[float] = Field(None, description="Area under curve to peak")
    post_peak_detected: bool = Field(False, description="Whether post-peak behavior captured")


class ConfidenceFactorsResponse(BaseModel):
    """Individual confidence factor scores."""
    ocr_confidence: float = Field(..., ge=0, le=1)
    curve_smoothness: float = Field(..., ge=0, le=1)
    axis_detection: float = Field(..., ge=0, le=1)
    image_quality: float = Field(..., ge=0, le=1)
    data_validity: float = Field(..., ge=0, le=1)
    extraction_method: float = Field(..., ge=0, le=1)


class ConfidenceResponse(BaseModel):
    """Complete confidence assessment."""
    overall_score: float = Field(..., ge=0, le=1, description="Overall confidence (0-1)")
    grade: str = Field(..., description="Letter grade (A-F)")
    factors: ConfidenceFactorsResponse
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class AxisInfoResponse(BaseModel):
    """Detection axis information."""
    label: Optional[str] = Field(None, description="Axis label text")
    unit: Optional[str] = Field(None, description="Axis unit")
    min_value: Optional[float] = Field(None, description="Minimum axis value")
    max_value: Optional[float] = Field(None, description="Maximum axis value")
    tick_count: int = Field(0, description="Number of detected tick marks")


class ExtractionRequest(BaseModel):
    """Request for graph extraction."""
    method: ExtractionMethod = Field(
        ExtractionMethod.AUTO,
        description="Extraction method to use"
    )
    expected_strain_range: Optional[tuple[float, float]] = Field(
        None,
        description="Expected strain range (min, max) for validation"
    )
    expected_stress_range: Optional[tuple[float, float]] = Field(
        None,
        description="Expected stress range (min, max) for validation"
    )
    apply_smoothing: bool = Field(
        True,
        description="Apply Savitzky-Golay smoothing"
    )
    smoothing_window: int = Field(
        11,
        ge=5,
        le=51,
        description="Smoothing window size (odd number)"
    )


class ExtractionResponse(BaseModel):
    """Response from extraction endpoint."""
    job_id: str = Field(..., description="Unique job identifier")
    status: JobStatus = Field(..., description="Current job status")
    data_points: List[DataPoint] = Field(
        default_factory=list,
        description="Extracted stress-strain data points"
    )
    features: Optional[UCSFeatureResponse] = Field(
        None,
        description="Extracted UCS features"
    )
    confidence: Optional[ConfidenceResponse] = Field(
        None,
        description="Confidence assessment"
    )
    x_axis: Optional[AxisInfoResponse] = Field(
        None,
        description="X-axis (strain) information"
    )
    y_axis: Optional[AxisInfoResponse] = Field(
        None,
        description="Y-axis (stress) information"
    )
    extraction_method: Optional[str] = Field(
        None,
        description="Method used for extraction"
    )
    processing_time_ms: Optional[float] = Field(
        None,
        description="Total processing time in milliseconds"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Job creation timestamp"
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Job completion timestamp"
    )
    error_message: Optional[str] = Field(
        None,
        description="Error message if failed"
    )


class ExtractionJob(BaseModel):
    """Internal job tracking model."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = Field(JobStatus.PENDING)
    filename: Optional[str] = None
    file_size: Optional[int] = None
    request: Optional[ExtractionRequest] = None
    result: Optional[ExtractionResponse] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ValidationRequest(BaseModel):
    """Request for validating extraction against ground truth."""
    extracted_data: List[DataPoint] = Field(
        ...,
        description="Extracted data points to validate"
    )
    ground_truth: List[DataPoint] = Field(
        ...,
        description="Ground truth data points from CSV"
    )


class ValidationMetrics(BaseModel):
    """Validation metrics."""
    mae: float = Field(..., description="Mean Absolute Error")
    rmse: float = Field(..., description="Root Mean Square Error")
    r_squared: float = Field(..., description="R² (coefficient of determination)")
    max_error: float = Field(..., description="Maximum point-wise error")
    peak_stress_error: float = Field(..., description="Error in peak stress detection")
    peak_strain_error: float = Field(..., description="Error in failure strain detection")
    within_5_percent: float = Field(..., description="Percentage of points within 5% error")
    within_10_percent: float = Field(..., description="Percentage of points within 10% error")


class ValidationResponse(BaseModel):
    """Response from validation endpoint."""
    is_valid: bool = Field(..., description="Whether extraction meets accuracy target")
    accuracy_score: float = Field(..., ge=0, le=1, description="Overall accuracy score")
    metrics: ValidationMetrics
    target_accuracy: float = Field(0.95, description="Target accuracy threshold")
    passed_target: bool = Field(..., description="Whether target accuracy is met")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class CSVExportRequest(BaseModel):
    """Request for CSV export configuration."""
    include_pixel_coords: bool = Field(False, description="Include pixel coordinates")
    include_smoothed: bool = Field(True, description="Include smoothed data")
    decimal_places: int = Field(4, ge=1, le=10, description="Decimal precision")
    separator: str = Field(",", description="CSV separator character")


class ImageUploadResponse(BaseModel):
    """Response after image upload."""
    job_id: str = Field(..., description="Created job ID")
    filename: str = Field(..., description="Uploaded filename")
    file_size: int = Field(..., description="File size in bytes")
    message: str = Field("Image uploaded successfully")
