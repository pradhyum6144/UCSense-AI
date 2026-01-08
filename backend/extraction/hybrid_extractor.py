"""
Hybrid Extractor Module

Intelligent routing between contour tracing and U-Net segmentation
based on image quality assessment.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

from config import settings
from .contour_tracer import ContourTracer, ExtractionResult
from .unet_segmenter import UNetSegmenter, SegmentationResult

logger = logging.getLogger(__name__)


@dataclass
class ImageQualityMetrics:
    """Image quality assessment metrics."""
    blur_score: float  # 0-1, higher = more blur
    noise_score: float  # 0-1, higher = more noise
    contrast_score: float  # 0-1, higher = better contrast
    overall_score: float  # 0-1, higher = better quality
    recommended_method: str  # "contour" or "unet"


@dataclass
class HybridExtractionResult:
    """Result of hybrid extraction."""
    points: List[Tuple[int, int]]
    smoothed_points: List[Tuple[int, int]]
    confidence: float
    method_used: str
    quality_metrics: ImageQualityMetrics
    contour_result: Optional[ExtractionResult] = None
    unet_result: Optional[SegmentationResult] = None


class HybridExtractor:
    """
    Hybrid curve extraction using both contour tracing and U-Net.
    
    Automatically selects the best method based on image quality
    assessment, or uses both and combines results.
    """
    
    def __init__(
        self,
        quality_threshold: float = None,
        always_try_both: bool = False
    ):
        """
        Initialize the hybrid extractor.
        
        Args:
            quality_threshold: Quality threshold for method selection
            always_try_both: If True, always run both methods
        """
        self.quality_threshold = quality_threshold or settings.image_quality_threshold
        self.always_try_both = always_try_both
        
        self.contour_tracer = ContourTracer()
        self.unet_segmenter = UNetSegmenter(
            model_path=settings.unet_model_path
        )
    
    def extract(
        self,
        image: np.ndarray,
        binary_mask: Optional[np.ndarray] = None,
        plot_region: Optional[Tuple[int, int, int, int]] = None
    ) -> HybridExtractionResult:
        """
        Extract curve using the optimal method.
        
        Args:
            image: Input BGR image
            binary_mask: Optional pre-computed binary mask
            plot_region: Optional plot region restriction
            
        Returns:
            HybridExtractionResult with extracted curve
        """
        # Assess image quality
        quality = self.assess_quality(image)
        logger.info(
            f"Image quality: {quality.overall_score:.2f}, "
            f"recommended: {quality.recommended_method}"
        )
        
        contour_result = None
        unet_result = None
        
        if self.always_try_both:
            # Try both methods
            contour_result = self._try_contour(image, binary_mask, plot_region)
            unet_result = self._try_unet(image)
            
            # Select best result
            points, smoothed, confidence, method = self._select_best(
                contour_result, unet_result
            )
        else:
            # Use quality-based selection
            if quality.overall_score >= self.quality_threshold:
                contour_result = self._try_contour(image, binary_mask, plot_region)
                
                if contour_result.confidence >= 0.7:
                    points = contour_result.points
                    smoothed = contour_result.smoothed_points
                    confidence = contour_result.confidence
                    method = "contour_tracing"
                else:
                    # Fall back to U-Net
                    unet_result = self._try_unet(image)
                    points = unet_result.points
                    smoothed = unet_result.points  # U-Net doesn't do separate smoothing
                    confidence = unet_result.confidence
                    method = "unet_fallback"
            else:
                # Low quality image, use U-Net directly
                unet_result = self._try_unet(image)
                points = unet_result.points
                smoothed = unet_result.points
                confidence = unet_result.confidence
                method = "unet_primary"
        
        return HybridExtractionResult(
            points=points,
            smoothed_points=smoothed,
            confidence=confidence,
            method_used=method,
            quality_metrics=quality,
            contour_result=contour_result,
            unet_result=unet_result
        )
    
    def assess_quality(self, image: np.ndarray) -> ImageQualityMetrics:
        """
        Assess the quality of an input image.
        
        Args:
            image: Input BGR image
            
        Returns:
            ImageQualityMetrics with quality scores
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate blur score using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        blur_variance = laplacian.var()
        # Normalize: higher variance = less blur = better
        blur_score = 1.0 - min(blur_variance / 500.0, 1.0)
        
        # Calculate noise score using local variance
        local_mean = cv2.blur(gray, (5, 5))
        local_var = cv2.blur((gray.astype(float) - local_mean.astype(float))**2, (5, 5))
        avg_local_var = np.mean(local_var)
        noise_score = min(avg_local_var / 1000.0, 1.0)
        
        # Calculate contrast score
        contrast = gray.std()
        contrast_score = min(contrast / 60.0, 1.0)
        
        # Overall quality score
        overall_score = (
            0.35 * (1.0 - blur_score) +  # Less blur is better
            0.35 * (1.0 - noise_score) +  # Less noise is better
            0.30 * contrast_score  # More contrast is better
        )
        
        # Determine recommended method
        if overall_score >= self.quality_threshold:
            recommended = "contour"
        else:
            recommended = "unet"
        
        return ImageQualityMetrics(
            blur_score=blur_score,
            noise_score=noise_score,
            contrast_score=contrast_score,
            overall_score=overall_score,
            recommended_method=recommended
        )
    
    def _try_contour(
        self,
        image: np.ndarray,
        binary_mask: Optional[np.ndarray],
        plot_region: Optional[Tuple[int, int, int, int]]
    ) -> ExtractionResult:
        """
        Attempt contour-based extraction.
        
        Args:
            image: Input image
            binary_mask: Optional binary mask
            plot_region: Optional plot region
            
        Returns:
            ExtractionResult
        """
        try:
            if binary_mask is not None:
                result = self.contour_tracer.extract(binary_mask, plot_region)
            else:
                result = self.contour_tracer.extract_from_image(image, plot_region)
            
            return result
        except Exception as e:
            logger.error(f"Contour extraction failed: {e}")
            return ExtractionResult(
                points=[],
                smoothed_points=[],
                confidence=0.0,
                method="contour_tracing_failed",
                contour_area=0.0,
                curve_length=0.0
            )
    
    def _try_unet(self, image: np.ndarray) -> SegmentationResult:
        """
        Attempt U-Net segmentation.
        
        Args:
            image: Input image
            
        Returns:
            SegmentationResult
        """
        try:
            return self.unet_segmenter.segment(image)
        except Exception as e:
            logger.error(f"U-Net segmentation failed: {e}")
            return SegmentationResult(
                mask=np.zeros(image.shape[:2], dtype=np.uint8),
                probability_map=np.zeros(image.shape[:2], dtype=np.float32),
                confidence=0.0,
                points=[]
            )
    
    def _select_best(
        self,
        contour_result: ExtractionResult,
        unet_result: SegmentationResult
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], float, str]:
        """
        Select the best result from contour and U-Net methods.
        
        Args:
            contour_result: Contour extraction result
            unet_result: U-Net segmentation result
            
        Returns:
            Tuple of (points, smoothed_points, confidence, method)
        """
        # Compare confidences
        contour_conf = contour_result.confidence if contour_result.points else 0.0
        unet_conf = unet_result.confidence if unet_result.points else 0.0
        
        logger.debug(
            f"Method comparison: contour={contour_conf:.2f}, unet={unet_conf:.2f}"
        )
        
        if contour_conf >= unet_conf and contour_conf > 0:
            return (
                contour_result.points,
                contour_result.smoothed_points,
                contour_conf,
                "contour_tracing"
            )
        elif unet_conf > 0:
            return (
                unet_result.points,
                unet_result.points,  # U-Net points are already processed
                unet_conf,
                "unet_segmentation"
            )
        else:
            # Both failed, return contour result anyway
            return (
                contour_result.points,
                contour_result.smoothed_points,
                0.0,
                "extraction_failed"
            )
    
    def combine_results(
        self,
        contour_result: ExtractionResult,
        unet_result: SegmentationResult
    ) -> List[Tuple[int, int]]:
        """
        Combine results from both methods for improved accuracy.
        
        Uses weighted averaging based on confidence scores.
        
        Args:
            contour_result: Contour extraction result
            unet_result: U-Net segmentation result
            
        Returns:
            Combined curve points
        """
        if not contour_result.points:
            return unet_result.points
        if not unet_result.points:
            return contour_result.points
        
        # Weight by confidence
        contour_weight = contour_result.confidence
        unet_weight = unet_result.confidence
        total_weight = contour_weight + unet_weight
        
        if total_weight == 0:
            return contour_result.points
        
        # Normalize weights
        contour_weight /= total_weight
        unet_weight /= total_weight
        
        # Create lookup from x to y for both results
        contour_dict = {x: y for x, y in contour_result.points}
        unet_dict = {x: y for x, y in unet_result.points}
        
        # Get all x values
        all_x = sorted(set(contour_dict.keys()) | set(unet_dict.keys()))
        
        # Combine points
        combined = []
        for x in all_x:
            contour_y = contour_dict.get(x)
            unet_y = unet_dict.get(x)
            
            if contour_y is not None and unet_y is not None:
                # Weighted average
                y = int(contour_weight * contour_y + unet_weight * unet_y)
            elif contour_y is not None:
                y = contour_y
            else:
                y = unet_y
            
            combined.append((x, y))
        
        return combined
