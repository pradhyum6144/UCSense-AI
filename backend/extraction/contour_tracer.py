"""
Contour Tracer Module

OpenCV-based curve extraction using contour detection for clean images.
Implements edge detection, contour finding, and spline interpolation.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of curve extraction."""
    points: List[Tuple[int, int]]  # Pixel coordinates
    smoothed_points: List[Tuple[int, int]]  # After smoothing
    confidence: float  # Extraction confidence (0-1)
    method: str  # Extraction method used
    contour_area: float
    curve_length: float


class ContourTracer:
    """
    Extracts UCS curve from clean graph images using contour detection.
    
    Best suited for high-quality scans with clear curve lines and
    minimal noise or artifacts.
    """
    
    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        min_contour_length: int = 100,
        simplify_epsilon: float = 1.0,
        smoothing_factor: float = 0.0
    ):
        """
        Initialize the contour tracer.
        
        Args:
            canny_low: Lower threshold for Canny edge detection
            canny_high: Upper threshold for Canny edge detection
            min_contour_length: Minimum contour arc length to consider
            simplify_epsilon: Epsilon for Douglas-Peucker simplification
            smoothing_factor: Smoothing factor for spline interpolation
        """
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_contour_length = min_contour_length
        self.simplify_epsilon = simplify_epsilon
        self.smoothing_factor = smoothing_factor
    
    def extract(
        self,
        binary_mask: np.ndarray,
        plot_region: Optional[Tuple[int, int, int, int]] = None
    ) -> ExtractionResult:
        """
        Extract the UCS curve from a binary mask.
        
        Args:
            binary_mask: Binary image with curve in white
            plot_region: Optional (x, y, w, h) to restrict extraction
            
        Returns:
            ExtractionResult with extracted curve points
        """
        # Apply region mask if provided
        if plot_region:
            x, y, w, h = plot_region
            mask = np.zeros_like(binary_mask)
            mask[y:y+h, x:x+w] = binary_mask[y:y+h, x:x+w]
            binary_mask = mask
        
        # Find contours
        contours, hierarchy = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )
        
        if not contours:
            logger.warning("No contours found in binary mask")
            return ExtractionResult(
                points=[],
                smoothed_points=[],
                confidence=0.0,
                method="contour_tracing",
                contour_area=0.0,
                curve_length=0.0
            )
        
        # Select the best contour (the UCS curve)
        best_contour = self._select_curve_contour(contours, binary_mask.shape)
        
        if best_contour is None:
            return ExtractionResult(
                points=[],
                smoothed_points=[],
                confidence=0.0,
                method="contour_tracing",
                contour_area=0.0,
                curve_length=0.0
            )
        
        # Simplify contour
        simplified = cv2.approxPolyDP(
            best_contour,
            self.simplify_epsilon,
            closed=False
        )
        
        # Extract points
        points = [(int(p[0][0]), int(p[0][1])) for p in simplified]
        
        # Sort by x-coordinate
        points.sort(key=lambda p: p[0])
        
        # Remove duplicates with same x
        points = self._remove_duplicate_x(points)
        
        # Apply spline smoothing
        smoothed_points = self._smooth_curve(points)
        
        # Calculate metrics
        contour_area = cv2.contourArea(best_contour)
        curve_length = cv2.arcLength(best_contour, closed=False)
        
        # Calculate confidence based on curve quality
        confidence = self._calculate_confidence(
            points, smoothed_points, binary_mask.shape
        )
        
        logger.info(
            f"Contour extraction: {len(points)} points, "
            f"confidence={confidence:.2f}"
        )
        
        return ExtractionResult(
            points=points,
            smoothed_points=smoothed_points,
            confidence=confidence,
            method="contour_tracing",
            contour_area=contour_area,
            curve_length=curve_length
        )
    
    def extract_from_image(
        self,
        image: np.ndarray,
        plot_region: Optional[Tuple[int, int, int, int]] = None
    ) -> ExtractionResult:
        """
        Extract curve directly from BGR image using edge detection.
        
        Args:
            image: BGR input image
            plot_region: Optional plot region to restrict extraction
            
        Returns:
            ExtractionResult with extracted curve points
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        
        # Dilate slightly to connect broken edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Use the binary edges as the mask
        return self.extract(edges, plot_region)
    
    def _select_curve_contour(
        self,
        contours: List[np.ndarray],
        image_shape: Tuple[int, int]
    ) -> Optional[np.ndarray]:
        """
        Select the contour most likely to be the UCS curve.
        
        Uses heuristics based on typical UCS curve characteristics:
        - Spans significant horizontal distance
        - Located in upper portion of plot area
        - Reasonably smooth and continuous
        
        Args:
            contours: List of detected contours
            image_shape: Shape of the source image
            
        Returns:
            Best matching contour or None
        """
        height, width = image_shape
        
        scored_contours = []
        
        for contour in contours:
            # Skip very short contours
            arc_length = cv2.arcLength(contour, closed=False)
            if arc_length < self.min_contour_length:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate scores
            # 1. Horizontal span (should cover significant width)
            width_score = w / width
            
            # 2. Position (UCS curves typically span from lower-left to upper area)
            # Lower y in image = higher on graph = higher stress
            position_score = 1.0 - (y + h / 2) / height
            
            # 3. Aspect ratio (curves are typically wider than tall)
            aspect_score = min(w / max(h, 1), 5) / 5
            
            # 4. Arc length relative to bounding box (smooth curves have longer paths)
            expected_length = np.sqrt(w**2 + h**2)
            smoothness_score = min(arc_length / max(expected_length, 1), 2) / 2
            
            # Combined score
            total_score = (
                0.35 * width_score +
                0.25 * position_score +
                0.20 * aspect_score +
                0.20 * smoothness_score
            )
            
            scored_contours.append((total_score, contour))
        
        if not scored_contours:
            return None
        
        # Return the highest-scoring contour
        scored_contours.sort(key=lambda x: x[0], reverse=True)
        return scored_contours[0][1]
    
    def _remove_duplicate_x(
        self,
        points: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Remove points with duplicate x-coordinates, keeping highest y value.
        
        This ensures a functional relationship x -> y for the curve.
        
        Args:
            points: List of (x, y) points
            
        Returns:
            Deduplicated points
        """
        if not points:
            return []
        
        # Group by x-coordinate
        x_to_y = {}
        for x, y in points:
            if x not in x_to_y:
                x_to_y[x] = []
            x_to_y[x].append(y)
        
        # Keep the median y for each x (robust to outliers)
        result = []
        for x in sorted(x_to_y.keys()):
            y_values = x_to_y[x]
            median_y = int(np.median(y_values))
            result.append((x, median_y))
        
        return result
    
    def _smooth_curve(
        self,
        points: List[Tuple[int, int]],
        num_output_points: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        """
        Apply spline smoothing to curve points.
        
        Args:
            points: Input points
            num_output_points: Number of output points (default: same as input)
            
        Returns:
            Smoothed points
        """
        if len(points) < 4:
            return points
        
        if num_output_points is None:
            num_output_points = len(points)
        
        try:
            x = np.array([p[0] for p in points])
            y = np.array([p[1] for p in points])
            
            # Apply Gaussian smoothing first
            y_smooth = gaussian_filter1d(y, sigma=2)
            
            # Fit spline
            tck, u = splprep([x, y_smooth], s=self.smoothing_factor, k=3)
            
            # Evaluate spline
            u_new = np.linspace(0, 1, num_output_points)
            x_new, y_new = splev(u_new, tck)
            
            return [(int(xi), int(yi)) for xi, yi in zip(x_new, y_new)]
            
        except Exception as e:
            logger.warning(f"Spline smoothing failed: {e}, using original points")
            return points
    
    def _calculate_confidence(
        self,
        points: List[Tuple[int, int]],
        smoothed_points: List[Tuple[int, int]],
        image_shape: Tuple[int, int]
    ) -> float:
        """
        Calculate extraction confidence based on curve quality.
        
        Args:
            points: Original extracted points
            smoothed_points: Smoothed points
            image_shape: Image dimensions
            
        Returns:
            Confidence score (0-1)
        """
        if not points or len(points) < 5:
            return 0.0
        
        height, width = image_shape
        
        # Factor 1: Point density
        min_x = min(p[0] for p in points)
        max_x = max(p[0] for p in points)
        span = max_x - min_x
        density = len(points) / max(span, 1)
        density_score = min(density * 10, 1.0)
        
        # Factor 2: Curve smoothness (deviation from smoothed)
        if len(smoothed_points) == len(points):
            deviations = [
                abs(p[1] - s[1])
                for p, s in zip(points, smoothed_points)
            ]
            avg_deviation = np.mean(deviations)
            smoothness_score = max(0, 1 - avg_deviation / 50)
        else:
            smoothness_score = 0.5
        
        # Factor 3: Horizontal coverage
        coverage = span / width
        coverage_score = min(coverage * 2, 1.0)
        
        # Factor 4: Monotonicity check (y should generally decrease as x increases)
        # until the peak, which is expected behavior for UCS curves
        mono_violations = 0
        for i in range(1, len(points)):
            # Allow small increases (noise) but penalize large ones
            if points[i][1] > points[i-1][1] + 10:
                mono_violations += 1
        mono_score = max(0, 1 - mono_violations / len(points))
        
        # Combined confidence
        confidence = (
            0.25 * density_score +
            0.30 * smoothness_score +
            0.25 * coverage_score +
            0.20 * mono_score
        )
        
        return float(confidence)
    
    def refine_extraction(
        self,
        result: ExtractionResult,
        original_image: np.ndarray
    ) -> ExtractionResult:
        """
        Refine extraction by fitting curve to detected edges more precisely.
        
        Args:
            result: Initial extraction result
            original_image: Original BGR image
            
        Returns:
            Refined ExtractionResult
        """
        if not result.points:
            return result
        
        # Convert to grayscale
        gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        
        refined_points = []
        
        for px, py in result.points:
            # Search in a small window around each point
            window_size = 5
            best_y = py
            best_intensity = 255
            
            for dy in range(-window_size, window_size + 1):
                new_y = py + dy
                if 0 <= new_y < gray.shape[0]:
                    intensity = gray[new_y, px]
                    # Look for darkest point (the curve)
                    if intensity < best_intensity:
                        best_intensity = intensity
                        best_y = new_y
            
            refined_points.append((px, best_y))
        
        # Re-smooth after refinement
        smoothed = self._smooth_curve(refined_points)
        
        # Recalculate confidence
        confidence = self._calculate_confidence(
            refined_points, smoothed, gray.shape
        )
        
        return ExtractionResult(
            points=refined_points,
            smoothed_points=smoothed,
            confidence=confidence,
            method="contour_tracing_refined",
            contour_area=result.contour_area,
            curve_length=result.curve_length
        )
