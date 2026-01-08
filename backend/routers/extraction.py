"""
Extraction Router

Main API endpoints for UCS graph extraction, processing, and export.
"""

import io
import csv
import time
import logging
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse

from config import settings
from models.schemas import (
    ExtractionRequest,
    ExtractionResponse,
    ExtractionJob,
    JobStatus,
    DataPoint,
    UCSFeatureResponse,
    ConfidenceResponse,
    ConfidenceFactorsResponse,
    AxisInfoResponse,
    ValidationRequest,
    ValidationResponse,
    ValidationMetrics,
    CSVExportRequest,
    ImageUploadResponse,
    ExtractionMethod
)
from preprocessing import ImageRectifier, MorphologicalFilter
from ocr import AxisDetector, ScaleMapper
from extraction import HybridExtractor
from analysis import PeakDetector, ConfidenceScorer

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory job storage (replace with database in production)
jobs: Dict[str, ExtractionJob] = {}


@router.post("/extract", response_model=ExtractionResponse)
async def extract_graph(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    method: ExtractionMethod = Query(ExtractionMethod.AUTO),
    apply_smoothing: bool = Query(True),
    smoothing_window: int = Query(11, ge=5, le=51)
):
    """
    Extract stress-strain data from a UCS graph image.
    
    Upload an image and receive digitized data points with confidence scoring.
    
    - **file**: Image file (PNG, JPG, TIFF)
    - **method**: Extraction method (auto, contour, unet, hybrid)
    - **apply_smoothing**: Whether to apply curve smoothing
    - **smoothing_window**: Savitzky-Golay window size
    """
    start_time = time.time()
    
    # Validate file type
    allowed_types = ["image/png", "image/jpeg", "image/tiff", "image/bmp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Read image
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
    except Exception as e:
        logger.error(f"Image read error: {e}")
        raise HTTPException(status_code=400, detail=f"Image read error: {str(e)}")
    
    # Create job
    request = ExtractionRequest(
        method=method,
        apply_smoothing=apply_smoothing,
        smoothing_window=smoothing_window
    )
    
    job = ExtractionJob(
        filename=file.filename,
        file_size=len(contents),
        request=request,
        status=JobStatus.PROCESSING,
        started_at=datetime.utcnow()
    )
    jobs[job.job_id] = job
    
    try:
        # Step 1: Image Rectification
        logger.info(f"Processing job {job.job_id}: Rectification")
        rectifier = ImageRectifier()
        rect_result = rectifier.rectify(image)
        rectified_image = rect_result.image
        
        # Step 2: Morphological Filtering
        logger.info(f"Processing job {job.job_id}: Filtering")
        filter_module = MorphologicalFilter()
        filter_result = filter_module.filter(rectified_image)
        
        # Step 3: Axis Detection
        logger.info(f"Processing job {job.job_id}: OCR")
        axis_detector = AxisDetector()
        axis_result = axis_detector.detect(rectified_image)
        
        # Step 4: Scale Mapping
        scale_mapper = ScaleMapper()
        scale_factors = scale_mapper.calculate_scale_factors(
            axis_result.x_axis,
            axis_result.y_axis,
            axis_result.origin_pixel
        )
        
        # Step 5: Curve Extraction
        logger.info(f"Processing job {job.job_id}: Extraction")
        extractor = HybridExtractor()
        extraction_result = extractor.extract(
            rectified_image,
            binary_mask=filter_result.binary_mask,
            plot_region=axis_result.plot_region
        )
        
        # Step 6: Map to Engineering Units
        if extraction_result.points and scale_factors.is_valid:
            mapped_points = scale_mapper.map_curve_to_units(
                extraction_result.smoothed_points or extraction_result.points,
                scale_factors,
                axis_result.origin_pixel
            )
            
            data_points = [
                DataPoint(
                    strain=p.strain,
                    stress=p.stress,
                    pixel_x=p.pixel_x,
                    pixel_y=p.pixel_y
                )
                for p in mapped_points
            ]
        else:
            # Fallback: use raw pixel coordinates if scale mapping failed
            data_points = [
                DataPoint(
                    strain=float(x),
                    stress=float(y),
                    pixel_x=x,
                    pixel_y=y
                )
                for x, y in (extraction_result.smoothed_points or extraction_result.points)
            ]
            logger.warning("Scale mapping failed, using pixel coordinates")
        
        # Step 7: Feature Extraction
        logger.info(f"Processing job {job.job_id}: Analysis")
        if data_points:
            peak_detector = PeakDetector(
                window_length=smoothing_window if apply_smoothing else 5
            )
            analysis = peak_detector.analyze(
                [p.strain for p in data_points],
                [p.stress for p in data_points]
            )
            
            features = UCSFeatureResponse(
                peak_stress=analysis.features.peak_stress,
                failure_strain=analysis.features.failure_strain,
                initial_modulus=analysis.features.initial_modulus,
                secant_modulus_50=analysis.features.secant_modulus_50,
                energy_to_peak=analysis.features.energy_to_peak,
                post_peak_detected=analysis.features.post_peak_detected
            )
        else:
            features = None
        
        # Step 8: Confidence Scoring
        confidence_scorer = ConfidenceScorer()
        
        ocr_confidences = [
            t.confidence for t in axis_result.x_axis.tick_marks
        ] + [
            t.confidence for t in axis_result.y_axis.tick_marks
        ]
        
        confidence_report = confidence_scorer.calculate_score(
            ocr_confidences=ocr_confidences,
            curve_points=[(p.strain, p.stress) for p in data_points],
            axis_tick_counts=(
                len(axis_result.x_axis.tick_marks),
                len(axis_result.y_axis.tick_marks)
            ),
            image_quality_score=extraction_result.quality_metrics.overall_score,
            extraction_method=extraction_result.method_used
        )
        
        confidence = ConfidenceResponse(
            overall_score=confidence_report.overall_score,
            grade=confidence_report.grade,
            factors=ConfidenceFactorsResponse(
                ocr_confidence=confidence_report.factors.ocr_confidence,
                curve_smoothness=confidence_report.factors.curve_smoothness,
                axis_detection=confidence_report.factors.axis_detection,
                image_quality=confidence_report.factors.image_quality,
                data_validity=confidence_report.factors.data_validity,
                extraction_method=confidence_report.factors.extraction_method
            ),
            warnings=confidence_report.warnings,
            recommendations=confidence_report.recommendations
        )
        
        # Build axis info
        x_axis_info = AxisInfoResponse(
            label=axis_result.x_axis.label,
            unit=axis_result.x_axis.unit or "%",
            min_value=axis_result.x_axis.tick_marks[0].value if axis_result.x_axis.tick_marks else None,
            max_value=axis_result.x_axis.tick_marks[-1].value if axis_result.x_axis.tick_marks else None,
            tick_count=len(axis_result.x_axis.tick_marks)
        )
        
        y_axis_info = AxisInfoResponse(
            label=axis_result.y_axis.label,
            unit=axis_result.y_axis.unit or "kN/m²",
            min_value=axis_result.y_axis.tick_marks[0].value if axis_result.y_axis.tick_marks else None,
            max_value=axis_result.y_axis.tick_marks[-1].value if axis_result.y_axis.tick_marks else None,
            tick_count=len(axis_result.y_axis.tick_marks)
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Build response
        response = ExtractionResponse(
            job_id=job.job_id,
            status=JobStatus.COMPLETED,
            data_points=data_points,
            features=features,
            confidence=confidence,
            x_axis=x_axis_info,
            y_axis=y_axis_info,
            extraction_method=extraction_result.method_used,
            processing_time_ms=processing_time,
            completed_at=datetime.utcnow()
        )
        
        # Update job
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.utcnow()
        job.result = response
        
        logger.info(
            f"Job {job.job_id} completed: {len(data_points)} points, "
            f"confidence={confidence.overall_score:.2f}, time={processing_time:.0f}ms"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Extraction failed for job {job.job_id}: {e}", exc_info=True)
        job.status = JobStatus.FAILED
        job.completed_at = datetime.utcnow()
        
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}"
        )


@router.get("/extract/{job_id}", response_model=ExtractionResponse)
async def get_extraction_result(job_id: str):
    """
    Get the results of a previous extraction job.
    
    - **job_id**: The job ID returned from the extract endpoint
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.result:
        return job.result
    
    # Return status if not completed
    return ExtractionResponse(
        job_id=job_id,
        status=job.status,
        created_at=job.created_at,
        error_message="Job not yet completed" if job.status != JobStatus.FAILED else "Extraction failed"
    )


@router.get("/extract/{job_id}/csv")
async def export_csv(
    job_id: str,
    include_pixel_coords: bool = Query(False),
    decimal_places: int = Query(4, ge=1, le=10)
):
    """
    Export extraction results as CSV.
    
    - **job_id**: The job ID
    - **include_pixel_coords**: Include pixel coordinates in output
    - **decimal_places**: Decimal precision for values
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job.status != JobStatus.COMPLETED or not job.result:
        raise HTTPException(status_code=400, detail="Job not completed")
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header
    if include_pixel_coords:
        writer.writerow(["Strain (%)", "Stress (kN/m²)", "Pixel_X", "Pixel_Y"])
    else:
        writer.writerow(["Strain (%)", "Stress (kN/m²)"])
    
    # Data rows
    fmt = f"{{:.{decimal_places}f}}"
    for point in job.result.data_points:
        if include_pixel_coords:
            writer.writerow([
                fmt.format(point.strain),
                fmt.format(point.stress),
                point.pixel_x or "",
                point.pixel_y or ""
            ])
        else:
            writer.writerow([
                fmt.format(point.strain),
                fmt.format(point.stress)
            ])
    
    # Return as streaming response
    output.seek(0)
    
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=ucs_data_{job_id[:8]}.csv"
        }
    )


@router.post("/validate", response_model=ValidationResponse)
async def validate_extraction(request: ValidationRequest):
    """
    Validate extracted data against ground truth CSV export.
    
    Calculates accuracy metrics including MAE, RMSE, and R².
    """
    extracted = request.extracted_data
    ground_truth = request.ground_truth
    
    if not extracted or not ground_truth:
        raise HTTPException(status_code=400, detail="Both datasets required")
    
    # Convert to numpy arrays for calculation
    ext_strains = np.array([p.strain for p in extracted])
    ext_stresses = np.array([p.stress for p in extracted])
    
    gt_strains = np.array([p.strain for p in ground_truth])
    gt_stresses = np.array([p.stress for p in ground_truth])
    
    # Interpolate to common points
    from scipy.interpolate import interp1d
    
    min_strain = max(ext_strains.min(), gt_strains.min())
    max_strain = min(ext_strains.max(), gt_strains.max())
    
    if min_strain >= max_strain:
        raise HTTPException(
            status_code=400,
            detail="No overlapping strain range between datasets"
        )
    
    common_strains = np.linspace(min_strain, max_strain, 100)
    
    ext_interp = interp1d(ext_strains, ext_stresses, fill_value='extrapolate')
    gt_interp = interp1d(gt_strains, gt_stresses, fill_value='extrapolate')
    
    ext_at_common = ext_interp(common_strains)
    gt_at_common = gt_interp(common_strains)
    
    # Calculate metrics
    errors = ext_at_common - gt_at_common
    abs_errors = np.abs(errors)
    
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    max_error = float(np.max(abs_errors))
    
    # R² calculation
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((gt_at_common - np.mean(gt_at_common)) ** 2)
    r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    # Normalize errors by max stress
    max_stress = np.max(gt_stresses)
    relative_errors = abs_errors / max_stress if max_stress > 0 else abs_errors
    
    within_5_percent = float(np.mean(relative_errors <= 0.05))
    within_10_percent = float(np.mean(relative_errors <= 0.10))
    
    # Peak detection errors
    ext_peak_idx = np.argmax(ext_stresses)
    gt_peak_idx = np.argmax(gt_stresses)
    
    peak_stress_error = float(abs(ext_stresses[ext_peak_idx] - gt_stresses[gt_peak_idx]))
    peak_strain_error = float(abs(ext_strains[ext_peak_idx] - gt_strains[gt_peak_idx]))
    
    # Normalize MAE for accuracy score
    normalized_mae = mae / max_stress if max_stress > 0 else 1.0
    accuracy_score = max(0, 1 - normalized_mae)
    
    target_accuracy = 0.95
    passed_target = accuracy_score >= target_accuracy
    
    metrics = ValidationMetrics(
        mae=mae,
        rmse=rmse,
        r_squared=r_squared,
        max_error=max_error,
        peak_stress_error=peak_stress_error,
        peak_strain_error=peak_strain_error,
        within_5_percent=within_5_percent,
        within_10_percent=within_10_percent
    )
    
    return ValidationResponse(
        is_valid=passed_target,
        accuracy_score=accuracy_score,
        metrics=metrics,
        target_accuracy=target_accuracy,
        passed_target=passed_target
    )


@router.delete("/extract/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its results."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del jobs[job_id]
    return {"message": "Job deleted successfully"}


@router.get("/jobs")
async def list_jobs(
    limit: int = Query(10, ge=1, le=100),
    status: Optional[JobStatus] = None
):
    """List recent extraction jobs."""
    job_list = list(jobs.values())
    
    if status:
        job_list = [j for j in job_list if j.status == status]
    
    # Sort by creation time (newest first)
    job_list.sort(key=lambda j: j.created_at, reverse=True)
    
    return {
        "jobs": [
            {
                "job_id": j.job_id,
                "status": j.status,
                "filename": j.filename,
                "created_at": j.created_at,
                "completed_at": j.completed_at
            }
            for j in job_list[:limit]
        ],
        "total": len(job_list)
    }
