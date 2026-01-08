"""
Axis Detection Module

Detects axis tick marks and labels from UCS graph images using
Tesseract OCR with intelligent validation and correction.
"""

import cv2
import numpy as np
import pytesseract
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import logging

from config import settings

logger = logging.getLogger(__name__)

# Configure Tesseract path if specified
if settings.tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


@dataclass
class TickMark:
    """Represents a detected tick mark on an axis."""
    pixel_position: int  # Position in pixels
    value: float  # Numeric value
    confidence: float  # OCR confidence (0-1)
    raw_text: str  # Original OCR text before correction


@dataclass
class AxisInfo:
    """Information about a detected axis."""
    tick_marks: List[TickMark] = field(default_factory=list)
    label: Optional[str] = None
    unit: Optional[str] = None
    is_valid: bool = False
    orientation: str = "horizontal"  # or "vertical"


@dataclass
class DetectionResult:
    """Complete axis detection result."""
    x_axis: AxisInfo
    y_axis: AxisInfo
    origin_pixel: Tuple[int, int]
    plot_region: Tuple[int, int, int, int]  # x, y, width, height


class AxisDetector:
    """
    Detects and reads axis information from graph images.
    
    Uses Tesseract OCR with custom preprocessing and validation
    to accurately read tick mark values and axis labels.
    """
    
    # Common OCR mistake corrections
    OCR_CORRECTIONS = {
        'O': '0', 'o': '0',
        'l': '1', 'I': '1', '|': '1',
        'S': '5', 's': '5',
        'B': '8',
        'Z': '2', 'z': '2',
        'G': '6', 'g': '9',
        ',': '.',  # Comma to decimal point
    }
    
    # Regex pattern for valid numeric values
    NUMERIC_PATTERN = re.compile(r'^-?\d+\.?\d*$')
    
    def __init__(
        self,
        min_confidence: float = 0.5,
        margin_ratio: float = 0.15
    ):
        """
        Initialize the axis detector.
        
        Args:
            min_confidence: Minimum OCR confidence to accept
            margin_ratio: Ratio of image to consider as axis margins
        """
        self.min_confidence = min_confidence
        self.margin_ratio = margin_ratio
    
    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Detect axes and read tick mark values.
        
        Args:
            image: Input BGR image
            
        Returns:
            DetectionResult with axis information
        """
        height, width = image.shape[:2]
        
        # Define regions for X and Y axes
        x_margin = int(width * self.margin_ratio)
        y_margin = int(height * self.margin_ratio)
        
        # X-axis region (bottom of image)
        x_axis_region = image[height - y_margin:, x_margin:]
        
        # Y-axis region (left side of image)
        y_axis_region = image[:height - y_margin, :x_margin]
        
        # Detect tick marks on each axis
        x_axis = self._detect_axis_ticks(x_axis_region, "horizontal")
        y_axis = self._detect_axis_ticks(y_axis_region, "vertical")
        
        # Detect axis labels
        x_axis.label, x_axis.unit = self._detect_axis_label(
            image[height - y_margin // 2:, :], "horizontal"
        )
        y_axis.label, y_axis.unit = self._detect_axis_label(
            image[:, :x_margin // 2], "vertical"
        )
        
        # Estimate origin position
        origin_x = x_margin
        origin_y = height - y_margin
        
        # Define plot region
        plot_region = (x_margin, 0, width - x_margin, height - y_margin)
        
        return DetectionResult(
            x_axis=x_axis,
            y_axis=y_axis,
            origin_pixel=(origin_x, origin_y),
            plot_region=plot_region
        )
    
    def _detect_axis_ticks(
        self,
        region: np.ndarray,
        orientation: str
    ) -> AxisInfo:
        """
        Detect tick marks in an axis region.
        
        Args:
            region: Image region containing the axis
            orientation: "horizontal" or "vertical"
            
        Returns:
            AxisInfo with detected tick marks
        """
        axis_info = AxisInfo(orientation=orientation)
        
        if region.size == 0:
            return axis_info
        
        # Preprocess for OCR
        processed = self._preprocess_for_ocr(region)
        
        # Run OCR with detailed output
        try:
            ocr_data = pytesseract.image_to_data(
                processed,
                lang=settings.tesseract_lang,
                output_type=pytesseract.Output.DICT,
                config='--psm 6 -c tessedit_char_whitelist=0123456789.-,'
            )
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            return axis_info
        
        # Process OCR results
        for i, text in enumerate(ocr_data['text']):
            if not text.strip():
                continue
            
            confidence = float(ocr_data['conf'][i]) / 100.0
            
            if confidence < self.min_confidence:
                continue
            
            # Correct common OCR mistakes
            corrected_text = self._correct_ocr_text(text)
            
            # Try to parse as number
            try:
                value = float(corrected_text)
            except ValueError:
                continue
            
            # Get position
            if orientation == "horizontal":
                position = ocr_data['left'][i] + ocr_data['width'][i] // 2
            else:
                position = ocr_data['top'][i] + ocr_data['height'][i] // 2
            
            tick_mark = TickMark(
                pixel_position=position,
                value=value,
                confidence=confidence,
                raw_text=text
            )
            
            axis_info.tick_marks.append(tick_mark)
        
        # Sort by position
        axis_info.tick_marks.sort(key=lambda t: t.pixel_position)
        
        # Validate axis (need at least 2 tick marks)
        axis_info.is_valid = len(axis_info.tick_marks) >= 2
        
        logger.debug(
            f"Detected {len(axis_info.tick_marks)} tick marks on {orientation} axis"
        )
        
        return axis_info
    
    def _preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image region for optimal OCR.
        
        Args:
            image: Input image region
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize for better OCR (Tesseract works best with larger text)
        scale = max(1, 300 // min(gray.shape))
        if scale > 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        
        # Apply bilateral filter to reduce noise while preserving edges
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Invert if needed (OCR works better with black text on white background)
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        
        return binary
    
    def _correct_ocr_text(self, text: str) -> str:
        """
        Correct common OCR mistakes in numeric text.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Corrected text
        """
        corrected = text.strip()
        
        # Apply character-level corrections
        for wrong, right in self.OCR_CORRECTIONS.items():
            corrected = corrected.replace(wrong, right)
        
        # Remove any remaining non-numeric characters (except . and -)
        corrected = re.sub(r'[^0-9.\-]', '', corrected)
        
        # Handle multiple decimal points (keep first)
        parts = corrected.split('.')
        if len(parts) > 2:
            corrected = parts[0] + '.' + ''.join(parts[1:])
        
        # Handle negative signs
        if corrected.count('-') > 1:
            if corrected.startswith('-'):
                corrected = '-' + corrected.replace('-', '')
            else:
                corrected = corrected.replace('-', '')
        
        return corrected
    
    def _detect_axis_label(
        self,
        region: np.ndarray,
        orientation: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Detect axis label and unit from region.
        
        Args:
            region: Image region containing axis label
            orientation: "horizontal" or "vertical"
            
        Returns:
            Tuple of (label, unit) or (None, None) if not detected
        """
        if region.size == 0:
            return None, None
        
        # For vertical axis, rotate for better OCR
        if orientation == "vertical":
            region = cv2.rotate(region, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Preprocess
        processed = self._preprocess_for_ocr(region)
        
        # Run OCR for text detection
        try:
            text = pytesseract.image_to_string(
                processed,
                lang=settings.tesseract_lang,
                config='--psm 7'
            ).strip()
        except Exception as e:
            logger.error(f"Label OCR failed: {e}")
            return None, None
        
        if not text:
            return None, None
        
        # Try to extract unit from common patterns
        unit_patterns = [
            r'\(([^)]+)\)',  # (unit)
            r'\[([^\]]+)\]',  # [unit]
            r'(kN/m[²2])',  # kN/m²
            r'(MPa)',
            r'(%)',
            r'(mm)',
        ]
        
        unit = None
        label = text
        
        for pattern in unit_patterns:
            match = re.search(pattern, text)
            if match:
                unit = match.group(1)
                label = text[:match.start()].strip()
                break
        
        return label if label else None, unit
    
    def refine_detection(
        self,
        result: DetectionResult,
        expected_x_range: Optional[Tuple[float, float]] = None,
        expected_y_range: Optional[Tuple[float, float]] = None
    ) -> DetectionResult:
        """
        Refine detection results using expected value ranges.
        
        This helps filter out obvious outliers and improve accuracy.
        
        Args:
            result: Initial detection result
            expected_x_range: Expected (min, max) for X-axis values
            expected_y_range: Expected (min, max) for Y-axis values
            
        Returns:
            Refined DetectionResult
        """
        if expected_x_range:
            result.x_axis.tick_marks = [
                t for t in result.x_axis.tick_marks
                if expected_x_range[0] <= t.value <= expected_x_range[1]
            ]
        
        if expected_y_range:
            result.y_axis.tick_marks = [
                t for t in result.y_axis.tick_marks
                if expected_y_range[0] <= t.value <= expected_y_range[1]
            ]
        
        # Re-validate after filtering
        result.x_axis.is_valid = len(result.x_axis.tick_marks) >= 2
        result.y_axis.is_valid = len(result.y_axis.tick_marks) >= 2
        
        return result
