"""
Confidence Scoring Module

Multi-factor confidence scoring system that combines OCR quality,
curve extraction accuracy, and data validation metrics.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import r2_score
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceFactors:
    """Individual confidence factors."""
    ocr_confidence: float
    curve_smoothness: float
    axis_detection: float
    image_quality: float
    data_validity: float
    extraction_method: float


@dataclass
class ConfidenceReport:
    """Complete confidence scoring report."""
    overall_score: float
    factors: ConfidenceFactors
    grade: str  # A, B, C, D, F
    warnings: List[str]
    recommendations: List[str]


class ConfidenceScorer:
    """
    Multi-factor confidence scoring for extraction quality.
    
    Combines multiple quality metrics to provide an overall
    confidence score and actionable recommendations.
    """
    
    # Grade thresholds
    GRADE_THRESHOLDS = {
        'A': 0.90,
        'B': 0.80,
        'C': 0.70,
        'D': 0.60,
        'F': 0.0
    }
    
    # Factor weights
    FACTOR_WEIGHTS = {
        'ocr_confidence': 0.20,
        'curve_smoothness': 0.25,
        'axis_detection': 0.15,
        'image_quality': 0.15,
        'data_validity': 0.15,
        'extraction_method': 0.10
    }
    
    def __init__(
        self,
        target_accuracy: float = 0.95,
        strict_mode: bool = False
    ):
        """
        Initialize the confidence scorer.
        
        Args:
            target_accuracy: Target accuracy threshold (for warnings)
            strict_mode: If True, apply stricter thresholds
        """
        self.target_accuracy = target_accuracy
        self.strict_mode = strict_mode
    
    def calculate_score(
        self,
        ocr_confidences: List[float],
        curve_points: List[Tuple[float, float]],
        axis_tick_counts: Tuple[int, int],
        image_quality_score: float,
        extraction_method: str,
        ground_truth: Optional[List[Tuple[float, float]]] = None
    ) -> ConfidenceReport:
        """
        Calculate comprehensive confidence score.
        
        Args:
            ocr_confidences: List of OCR confidence values (0-1)
            curve_points: Extracted (strain, stress) points
            axis_tick_counts: (x_ticks, y_ticks) counts
            image_quality_score: Image quality assessment (0-1)
            extraction_method: Method used ("contour", "unet", "hybrid")
            ground_truth: Optional ground truth data for validation
            
        Returns:
            ConfidenceReport with overall score and details
        """
        warnings = []
        recommendations = []
        
        # Calculate individual factors
        
        # 1. OCR Confidence
        if ocr_confidences:
            ocr_conf = float(np.mean(ocr_confidences))
            if ocr_conf < 0.7:
                warnings.append("Low OCR confidence on axis labels")
                recommendations.append("Consider manual verification of axis values")
        else:
            ocr_conf = 0.0
            warnings.append("No OCR data available")
        
        # 2. Curve Smoothness
        if len(curve_points) >= 3:
            curve_smooth = self._calculate_smoothness(curve_points)
            if curve_smooth < 0.7:
                warnings.append("Extracted curve appears jagged")
                recommendations.append("Apply additional smoothing or re-extract")
        else:
            curve_smooth = 0.0
            warnings.append("Insufficient curve points")
        
        # 3. Axis Detection Quality
        x_ticks, y_ticks = axis_tick_counts
        axis_score = self._calculate_axis_score(x_ticks, y_ticks)
        if axis_score < 0.7:
            warnings.append("Limited axis tick marks detected")
            recommendations.append("Verify scale mapping manually")
        
        # 4. Image Quality
        img_quality = image_quality_score
        if img_quality < 0.6:
            warnings.append("Low image quality may affect accuracy")
            recommendations.append("Use higher resolution scan if available")
        
        # 5. Data Validity
        if curve_points:
            data_valid = self._validate_data(curve_points)
            if data_valid < 0.7:
                warnings.append("Extracted data shows anomalies")
        else:
            data_valid = 0.0
        
        # 6. Extraction Method Score
        method_scores = {
            "contour_tracing": 0.9,
            "contour_tracing_refined": 0.95,
            "unet_segmentation": 0.85,
            "unet_primary": 0.85,
            "unet_fallback": 0.75,
            "hybrid": 0.9,
            "extraction_failed": 0.0
        }
        method_score = method_scores.get(extraction_method, 0.5)
        
        # Compare to ground truth if available
        if ground_truth and curve_points:
            mae = self._calculate_mae(curve_points, ground_truth)
            if mae > 0.05:  # More than 5% error
                data_valid *= (1 - min(mae, 0.5))
                warnings.append(f"MAE against ground truth: {mae:.1%}")
        
        # Create factors dataclass
        factors = ConfidenceFactors(
            ocr_confidence=ocr_conf,
            curve_smoothness=curve_smooth,
            axis_detection=axis_score,
            image_quality=img_quality,
            data_validity=data_valid,
            extraction_method=method_score
        )
        
        # Calculate weighted overall score
        overall = (
            self.FACTOR_WEIGHTS['ocr_confidence'] * factors.ocr_confidence +
            self.FACTOR_WEIGHTS['curve_smoothness'] * factors.curve_smoothness +
            self.FACTOR_WEIGHTS['axis_detection'] * factors.axis_detection +
            self.FACTOR_WEIGHTS['image_quality'] * factors.image_quality +
            self.FACTOR_WEIGHTS['data_validity'] * factors.data_validity +
            self.FACTOR_WEIGHTS['extraction_method'] * factors.extraction_method
        )
        
        # Apply strict mode penalty
        if self.strict_mode:
            min_factor = min(
                factors.ocr_confidence,
                factors.curve_smoothness,
                factors.axis_detection,
                factors.data_validity
            )
            if min_factor < 0.5:
                overall *= 0.8
        
        # Determine grade
        grade = 'F'
        for g, threshold in self.GRADE_THRESHOLDS.items():
            if overall >= threshold:
                grade = g
                break
        
        # Add grade-specific recommendations
        if grade in ['D', 'F']:
            recommendations.append("Manual review strongly recommended")
            recommendations.append("Consider re-scanning the original document")
        elif grade == 'C':
            recommendations.append("Verify critical values before use")
        
        # Check against target accuracy
        if overall < self.target_accuracy:
            accuracy_gap = self.target_accuracy - overall
            warnings.append(
                f"Below target accuracy ({self.target_accuracy:.0%}) by {accuracy_gap:.1%}"
            )
        
        return ConfidenceReport(
            overall_score=overall,
            factors=factors,
            grade=grade,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _calculate_smoothness(
        self,
        points: List[Tuple[float, float]]
    ) -> float:
        """
        Calculate curve smoothness score.
        
        Uses second derivative to measure how smooth the curve is.
        
        Args:
            points: List of (strain, stress) points
            
        Returns:
            Smoothness score (0-1)
        """
        if len(points) < 5:
            return 0.5
        
        strains = np.array([p[0] for p in points])
        stresses = np.array([p[1] for p in points])
        
        # Calculate second derivative
        first_deriv = np.gradient(stresses, strains)
        second_deriv = np.gradient(first_deriv, strains)
        
        # Smoothness is inversely related to second derivative variance
        second_deriv_var = np.var(second_deriv)
        
        # Normalize (empirically determined)
        normalized_var = second_deriv_var / (np.var(stresses) + 1e-6)
        
        # Convert to 0-1 score (lower variance = higher smoothness)
        smoothness = 1.0 / (1.0 + normalized_var * 10)
        
        return float(min(smoothness, 1.0))
    
    def _calculate_axis_score(
        self,
        x_ticks: int,
        y_ticks: int
    ) -> float:
        """
        Calculate axis detection quality score.
        
        Args:
            x_ticks: Number of detected X-axis tick marks
            y_ticks: Number of detected Y-axis tick marks
            
        Returns:
            Axis detection score (0-1)
        """
        # Ideal is 5+ ticks on each axis
        x_score = min(x_ticks / 5, 1.0)
        y_score = min(y_ticks / 5, 1.0)
        
        # Both axes need to be good
        combined = (x_score + y_score) / 2
        
        # Penalty if either axis has too few ticks
        if x_ticks < 2 or y_ticks < 2:
            combined *= 0.5
        
        return float(combined)
    
    def _validate_data(
        self,
        points: List[Tuple[float, float]]
    ) -> float:
        """
        Validate extracted data for physical plausibility.
        
        Args:
            points: List of (strain, stress) points
            
        Returns:
            Data validity score (0-1)
        """
        if len(points) < 3:
            return 0.0
        
        strains = np.array([p[0] for p in points])
        stresses = np.array([p[1] for p in points])
        
        scores = []
        
        # Check 1: Strain values should be non-negative
        if np.any(strains < -0.1):
            scores.append(0.5)
        else:
            scores.append(1.0)
        
        # Check 2: Stress values should be non-negative (compression is positive)
        if np.any(stresses < -100):
            scores.append(0.5)
        else:
            scores.append(1.0)
        
        # Check 3: Generally increasing stress in initial portion
        initial_len = len(stresses) // 3
        if initial_len > 2:
            initial_increases = np.sum(np.diff(stresses[:initial_len]) > 0)
            initial_score = initial_increases / (initial_len - 1)
            scores.append(initial_score)
        
        # Check 4: Strain should be monotonically increasing
        strain_increases = np.sum(np.diff(strains) >= 0)
        strain_mono_score = strain_increases / (len(strains) - 1)
        scores.append(strain_mono_score)
        
        # Check 5: Data density (should have reasonable number of points)
        density_score = min(len(points) / 50, 1.0)
        scores.append(density_score)
        
        return float(np.mean(scores))
    
    def _calculate_mae(
        self,
        extracted: List[Tuple[float, float]],
        ground_truth: List[Tuple[float, float]]
    ) -> float:
        """
        Calculate Mean Absolute Error against ground truth.
        
        Args:
            extracted: Extracted data points
            ground_truth: Ground truth data points
            
        Returns:
            Normalized MAE (0-1)
        """
        if not extracted or not ground_truth:
            return 1.0
        
        # Create interpolation for comparison
        ext_strains = np.array([p[0] for p in extracted])
        ext_stresses = np.array([p[1] for p in extracted])
        
        gt_strains = np.array([p[0] for p in ground_truth])
        gt_stresses = np.array([p[1] for p in ground_truth])
        
        # Find overlapping strain range
        min_strain = max(ext_strains.min(), gt_strains.min())
        max_strain = min(ext_strains.max(), gt_strains.max())
        
        if min_strain >= max_strain:
            return 1.0
        
        # Compare at common strain points
        from scipy.interpolate import interp1d
        
        ext_interp = interp1d(ext_strains, ext_stresses, fill_value='extrapolate')
        gt_interp = interp1d(gt_strains, gt_stresses, fill_value='extrapolate')
        
        common_strains = np.linspace(min_strain, max_strain, 50)
        ext_at_common = ext_interp(common_strains)
        gt_at_common = gt_interp(common_strains)
        
        # Calculate normalized MAE
        mae = np.mean(np.abs(ext_at_common - gt_at_common))
        max_stress = np.max(gt_stresses)
        
        if max_stress > 0:
            normalized_mae = mae / max_stress
        else:
            normalized_mae = 1.0
        
        return float(normalized_mae)
    
    def get_improvement_suggestions(
        self,
        report: ConfidenceReport
    ) -> List[Dict[str, str]]:
        """
        Get detailed improvement suggestions based on confidence report.
        
        Args:
            report: Confidence report from calculate_score
            
        Returns:
            List of suggestion dictionaries with 'factor', 'issue', 'solution'
        """
        suggestions = []
        factors = report.factors
        
        if factors.ocr_confidence < 0.7:
            suggestions.append({
                'factor': 'OCR',
                'issue': f'Low OCR confidence ({factors.ocr_confidence:.0%})',
                'solution': 'Improve image contrast or manually input axis values'
            })
        
        if factors.curve_smoothness < 0.7:
            suggestions.append({
                'factor': 'Curve Quality',
                'issue': f'Jagged curve extraction ({factors.curve_smoothness:.0%})',
                'solution': 'Apply Savitzky-Golay smoothing or use U-Net extraction'
            })
        
        if factors.axis_detection < 0.7:
            suggestions.append({
                'factor': 'Axis Detection',
                'issue': f'Insufficient tick marks ({factors.axis_detection:.0%})',
                'solution': 'Manually specify axis range and scale'
            })
        
        if factors.image_quality < 0.6:
            suggestions.append({
                'factor': 'Image Quality',
                'issue': f'Poor image quality ({factors.image_quality:.0%})',
                'solution': 'Re-scan at higher resolution (300+ DPI)'
            })
        
        if factors.data_validity < 0.7:
            suggestions.append({
                'factor': 'Data Validity',
                'issue': f'Data anomalies detected ({factors.data_validity:.0%})',
                'solution': 'Review extracted data for outliers'
            })
        
        return suggestions
