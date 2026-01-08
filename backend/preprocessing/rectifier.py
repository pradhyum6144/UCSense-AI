"""
Image Rectification Module

Performs perspective correction and deskewing of UCS graph images
to ensure the coordinate system is perfectly orthogonal.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class RectificationResult:
    """Result of image rectification process."""
    image: np.ndarray
    rotation_angle: float
    was_corrected: bool
    original_shape: Tuple[int, int]
    corrected_shape: Tuple[int, int]


class ImageRectifier:
    """
    Performs perspective correction and deskewing on graph images.
    
    Uses Hough Line Transform to detect the predominant line orientations
    and applies appropriate transformations to align axes.
    """
    
    def __init__(
        self,
        angle_threshold: float = 0.5,
        min_line_length: int = 100,
        max_line_gap: int = 10
    ):
        """
        Initialize the image rectifier.
        
        Args:
            angle_threshold: Maximum acceptable skew angle (degrees)
            min_line_length: Minimum length for Hough line detection
            max_line_gap: Maximum gap between line segments
        """
        self.angle_threshold = angle_threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
    
    def rectify(self, image: np.ndarray) -> RectificationResult:
        """
        Perform complete image rectification.
        
        Args:
            image: Input BGR or grayscale image
            
        Returns:
            RectificationResult with corrected image and metadata
        """
        original_shape = image.shape[:2]
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect skew angle
        angle = self._detect_skew_angle(gray)
        logger.debug(f"Detected skew angle: {angle:.2f} degrees")
        
        # Apply deskewing if needed
        if abs(angle) > self.angle_threshold:
            corrected = self._deskew(image, angle)
            was_corrected = True
        else:
            corrected = image.copy()
            was_corrected = False
        
        # Apply perspective correction if needed
        corrected = self._correct_perspective(corrected)
        
        # Auto-crop to remove black borders
        corrected = self._auto_crop(corrected)
        
        return RectificationResult(
            image=corrected,
            rotation_angle=angle if was_corrected else 0.0,
            was_corrected=was_corrected,
            original_shape=original_shape,
            corrected_shape=corrected.shape[:2]
        )
    
    def _detect_skew_angle(self, gray: np.ndarray) -> float:
        """
        Detect the skew angle using Hough Line Transform.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Detected skew angle in degrees
        """
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )
        
        if lines is None or len(lines) == 0:
            return 0.0
        
        # Calculate angles of detected lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Focus on near-horizontal and near-vertical lines
            # (these represent the axes)
            if abs(angle) < 45:  # Near horizontal
                angles.append(angle)
            elif abs(angle) > 45:  # Near vertical
                # Normalize to horizontal reference
                angles.append(angle - 90 if angle > 0 else angle + 90)
        
        if not angles:
            return 0.0
        
        # Use median to be robust against outliers
        return float(np.median(angles))
    
    def _deskew(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate image to correct skew.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees
            
        Returns:
            Deskewed image
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Calculate rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new bounding box size
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust rotation matrix for translation
        rotation_matrix[0, 2] += (new_w - w) // 2
        rotation_matrix[1, 2] += (new_h - h) // 2
        
        # Apply rotation
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        return rotated
    
    def _correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """
        Apply perspective correction if needed.
        
        Detects the graph bounding box and applies homography
        to correct for camera angle distortion.
        
        Args:
            image: Input image
            
        Returns:
            Perspective-corrected image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Try to detect graph border/frame
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Find the largest contour (likely the graph border)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate to polygon
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # If we found a quadrilateral, apply perspective transform
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            
            # Order points: top-left, top-right, bottom-right, bottom-left
            pts = self._order_points(pts)
            
            # Calculate dimensions
            width = int(max(
                np.linalg.norm(pts[0] - pts[1]),
                np.linalg.norm(pts[2] - pts[3])
            ))
            height = int(max(
                np.linalg.norm(pts[0] - pts[3]),
                np.linalg.norm(pts[1] - pts[2])
            ))
            
            # Destination points
            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
            
            # Apply perspective transform
            matrix = cv2.getPerspectiveTransform(pts, dst)
            corrected = cv2.warpPerspective(image, matrix, (width, height))
            
            return corrected
        
        return image
    
    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Order points in clockwise order starting from top-left.
        
        Args:
            pts: Array of 4 points
            
        Returns:
            Ordered points array
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum of coordinates: top-left has smallest, bottom-right has largest
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        # Difference of coordinates: top-right has smallest, bottom-left has largest
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]
        rect[3] = pts[np.argmax(d)]
        
        return rect
    
    def _auto_crop(self, image: np.ndarray) -> np.ndarray:
        """
        Auto-crop the image to remove black borders.
        
        Args:
            image: Input image
            
        Returns:
            Cropped image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Threshold to create mask
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        
        # Find bounding box of non-black region
        coords = cv2.findNonZero(thresh)
        
        if coords is None:
            return image
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add small padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        return image[y:y+h, x:x+w]
