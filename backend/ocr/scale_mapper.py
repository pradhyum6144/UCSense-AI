"""
Scale Mapper Module

Maps pixel coordinates to engineering units using detected axis
tick marks and linear regression for accurate scale calculation.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from sklearn.linear_model import RANSACRegressor, LinearRegression
import logging

from .axis_detector import AxisInfo, TickMark

logger = logging.getLogger(__name__)


@dataclass
class ScaleFactors:
    """Scale factors for coordinate transformation."""
    x_scale: float  # Units per pixel for X-axis
    y_scale: float  # Units per pixel for Y-axis
    x_offset: float  # X-axis offset (value at origin)
    y_offset: float  # Y-axis offset (value at origin)
    x_r_squared: float  # R² score for X-axis regression
    y_r_squared: float  # R² score for Y-axis regression
    is_valid: bool


@dataclass
class MappedPoint:
    """A point mapped to engineering coordinates."""
    pixel_x: int
    pixel_y: int
    strain: float  # Percentage
    stress: float  # kN/m² (or other unit)


class ScaleMapper:
    """
    Maps pixel coordinates to engineering units.
    
    Uses linear regression with RANSAC for robust outlier rejection
    to calculate accurate scale factors from detected tick marks.
    """
    
    def __init__(
        self,
        use_ransac: bool = True,
        ransac_residual_threshold: float = 5.0,
        min_tick_marks: int = 2
    ):
        """
        Initialize the scale mapper.
        
        Args:
            use_ransac: Whether to use RANSAC for outlier-robust regression
            ransac_residual_threshold: Max residual for RANSAC inliers
            min_tick_marks: Minimum tick marks required for valid mapping
        """
        self.use_ransac = use_ransac
        self.ransac_residual_threshold = ransac_residual_threshold
        self.min_tick_marks = min_tick_marks
    
    def calculate_scale_factors(
        self,
        x_axis: AxisInfo,
        y_axis: AxisInfo,
        origin_pixel: Tuple[int, int]
    ) -> ScaleFactors:
        """
        Calculate scale factors from axis tick marks.
        
        Args:
            x_axis: X-axis information with tick marks
            y_axis: Y-axis information with tick marks
            origin_pixel: Pixel coordinates of the origin
            
        Returns:
            ScaleFactors for coordinate transformation
        """
        origin_x, origin_y = origin_pixel
        
        # Calculate X-axis scale
        x_scale, x_offset, x_r2 = self._fit_scale(
            x_axis.tick_marks,
            is_horizontal=True,
            origin_pixel=origin_x
        )
        
        # Calculate Y-axis scale (note: Y increases downward in images)
        y_scale, y_offset, y_r2 = self._fit_scale(
            y_axis.tick_marks,
            is_horizontal=False,
            origin_pixel=origin_y
        )
        
        # Validate results
        is_valid = (
            x_r2 is not None and x_r2 > 0.9 and
            y_r2 is not None and y_r2 > 0.9 and
            x_scale is not None and x_scale > 0 and
            y_scale is not None
        )
        
        return ScaleFactors(
            x_scale=x_scale or 1.0,
            y_scale=y_scale or 1.0,
            x_offset=x_offset or 0.0,
            y_offset=y_offset or 0.0,
            x_r_squared=x_r2 or 0.0,
            y_r_squared=y_r2 or 0.0,
            is_valid=is_valid
        )
    
    def _fit_scale(
        self,
        tick_marks: List[TickMark],
        is_horizontal: bool,
        origin_pixel: int
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Fit a linear scale from tick marks using regression.
        
        Args:
            tick_marks: List of detected tick marks
            is_horizontal: Whether this is a horizontal (X) axis
            origin_pixel: Pixel position of the origin
            
        Returns:
            Tuple of (scale, offset, r_squared) or (None, None, None) if failed
        """
        if len(tick_marks) < self.min_tick_marks:
            logger.warning(
                f"Insufficient tick marks ({len(tick_marks)}) for scale fitting"
            )
            return None, None, None
        
        # Prepare data
        positions = np.array([t.pixel_position for t in tick_marks]).reshape(-1, 1)
        values = np.array([t.value for t in tick_marks])
        
        # Weight by confidence
        weights = np.array([t.confidence for t in tick_marks])
        
        try:
            if self.use_ransac and len(tick_marks) >= 3:
                # Use RANSAC for robust fitting
                ransac = RANSACRegressor(
                    estimator=LinearRegression(),
                    residual_threshold=self.ransac_residual_threshold,
                    random_state=42
                )
                ransac.fit(positions, values)
                
                # Get inlier mask
                inlier_mask = ransac.inlier_mask_
                n_inliers = np.sum(inlier_mask)
                
                logger.debug(
                    f"RANSAC: {n_inliers}/{len(tick_marks)} inliers"
                )
                
                # Calculate R² on inliers
                if n_inliers >= 2:
                    pred = ransac.predict(positions[inlier_mask])
                    r_squared = self._calculate_r_squared(
                        values[inlier_mask], pred
                    )
                else:
                    r_squared = 0.0
                
                slope = ransac.estimator_.coef_[0]
                intercept = ransac.estimator_.intercept_
                
            else:
                # Simple weighted linear regression
                reg = LinearRegression()
                reg.fit(positions, values, sample_weight=weights)
                
                pred = reg.predict(positions)
                r_squared = self._calculate_r_squared(values, pred)
                
                slope = reg.coef_[0]
                intercept = reg.intercept_
            
            # Calculate scale (units per pixel)
            scale = slope
            
            # For Y-axis, flip sign since image Y increases downward
            if not is_horizontal:
                scale = -scale
            
            # Calculate offset (value at origin pixel)
            offset = intercept + slope * origin_pixel
            
            logger.debug(
                f"{'X' if is_horizontal else 'Y'}-axis: "
                f"scale={scale:.6f}, offset={offset:.2f}, R²={r_squared:.4f}"
            )
            
            return float(scale), float(offset), float(r_squared)
            
        except Exception as e:
            logger.error(f"Scale fitting failed: {e}")
            return None, None, None
    
    def _calculate_r_squared(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> float:
        """
        Calculate R² (coefficient of determination).
        
        Args:
            y_true: Actual values
            y_pred: Predicted values
            
        Returns:
            R² score
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def map_pixel_to_units(
        self,
        pixel_x: int,
        pixel_y: int,
        scale_factors: ScaleFactors,
        origin_pixel: Tuple[int, int]
    ) -> MappedPoint:
        """
        Map a pixel coordinate to engineering units.
        
        Args:
            pixel_x: X pixel coordinate
            pixel_y: Y pixel coordinate
            scale_factors: Calculated scale factors
            origin_pixel: Pixel coordinates of the origin
            
        Returns:
            MappedPoint with engineering coordinates
        """
        origin_x, origin_y = origin_pixel
        
        # Calculate strain (X-axis)
        dx = pixel_x - origin_x
        strain = scale_factors.x_offset + dx * scale_factors.x_scale
        
        # Calculate stress (Y-axis) - note Y increases downward in images
        dy = origin_y - pixel_y
        stress = scale_factors.y_offset + dy * scale_factors.y_scale
        
        return MappedPoint(
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            strain=strain,
            stress=stress
        )
    
    def map_curve_to_units(
        self,
        pixel_points: List[Tuple[int, int]],
        scale_factors: ScaleFactors,
        origin_pixel: Tuple[int, int]
    ) -> List[MappedPoint]:
        """
        Map a series of pixel coordinates to engineering units.
        
        Args:
            pixel_points: List of (x, y) pixel coordinates
            scale_factors: Calculated scale factors
            origin_pixel: Pixel coordinates of the origin
            
        Returns:
            List of MappedPoints
        """
        mapped_points = []
        
        for px, py in pixel_points:
            point = self.map_pixel_to_units(px, py, scale_factors, origin_pixel)
            mapped_points.append(point)
        
        # Sort by strain (x-value)
        mapped_points.sort(key=lambda p: p.strain)
        
        return mapped_points
    
    def validate_mapping(
        self,
        mapped_points: List[MappedPoint],
        expected_strain_range: Tuple[float, float] = (0, 10),
        expected_stress_range: Tuple[float, float] = (0, 50000)
    ) -> Tuple[bool, List[str]]:
        """
        Validate mapped data against expected ranges.
        
        Args:
            mapped_points: List of mapped data points
            expected_strain_range: Expected (min, max) strain values
            expected_stress_range: Expected (min, max) stress values
            
        Returns:
            Tuple of (is_valid, list of warnings/errors)
        """
        issues = []
        
        if not mapped_points:
            return False, ["No data points to validate"]
        
        strains = [p.strain for p in mapped_points]
        stresses = [p.stress for p in mapped_points]
        
        min_strain, max_strain = min(strains), max(strains)
        min_stress, max_stress = min(stresses), max(stresses)
        
        # Check strain range
        if min_strain < expected_strain_range[0]:
            issues.append(
                f"Strain minimum ({min_strain:.2f}) below expected ({expected_strain_range[0]})"
            )
        if max_strain > expected_strain_range[1]:
            issues.append(
                f"Strain maximum ({max_strain:.2f}) above expected ({expected_strain_range[1]})"
            )
        
        # Check stress range
        if min_stress < expected_stress_range[0]:
            issues.append(
                f"Stress minimum ({min_stress:.2f}) below expected ({expected_stress_range[0]})"
            )
        if max_stress > expected_stress_range[1]:
            issues.append(
                f"Stress maximum ({max_stress:.2f}) above expected ({expected_stress_range[1]})"
            )
        
        # Check for monotonicity in early portion (strain should increase)
        early_points = mapped_points[:len(mapped_points) // 4]
        if len(early_points) > 1:
            strain_diffs = [
                early_points[i+1].strain - early_points[i].strain
                for i in range(len(early_points) - 1)
            ]
            if any(d < 0 for d in strain_diffs):
                issues.append("Non-monotonic strain detected in early curve portion")
        
        is_valid = len(issues) == 0
        
        return is_valid, issues
