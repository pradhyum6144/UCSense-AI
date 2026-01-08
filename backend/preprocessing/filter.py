"""
Morphological Filtering Module

Implements noise filtering to isolate UCS curves from distracting
background elements like grid lines, watermarks, and annotations.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of morphological filtering process."""
    filtered_image: np.ndarray
    binary_mask: np.ndarray
    grid_removed: bool
    noise_level: float


class MorphologicalFilter:
    """
    Signal-from-noise extraction using morphological operations.
    
    Removes grid lines, watermarks, and annotations while preserving
    the UCS curve signal.
    """
    
    def __init__(
        self,
        blur_kernel_size: int = 5,
        adaptive_block_size: int = 11,
        adaptive_c: int = 2,
        grid_kernel_size: int = 15
    ):
        """
        Initialize the morphological filter.
        
        Args:
            blur_kernel_size: Kernel size for Gaussian blur
            adaptive_block_size: Block size for adaptive thresholding
            adaptive_c: Constant subtracted from mean in adaptive threshold
            grid_kernel_size: Kernel size for grid line removal
        """
        self.blur_kernel_size = blur_kernel_size
        self.adaptive_block_size = adaptive_block_size
        self.adaptive_c = adaptive_c
        self.grid_kernel_size = grid_kernel_size
    
    def filter(self, image: np.ndarray) -> FilterResult:
        """
        Apply complete morphological filtering pipeline.
        
        Args:
            image: Input BGR or grayscale image
            
        Returns:
            FilterResult with cleaned image and metadata
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate initial noise level
        noise_level = self._estimate_noise(gray)
        logger.debug(f"Estimated noise level: {noise_level:.4f}")
        
        # Step 1: Denoise with Gaussian blur
        denoised = cv2.GaussianBlur(
            gray,
            (self.blur_kernel_size, self.blur_kernel_size),
            0
        )
        
        # Step 2: Remove grid lines
        grid_removed_img, grid_was_removed = self._remove_grid_lines(denoised)
        
        # Step 3: Remove watermarks if present
        cleaned = self._remove_watermarks(grid_removed_img)
        
        # Step 4: Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            cleaned,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.adaptive_block_size,
            self.adaptive_c
        )
        
        # Step 5: Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Opening to remove small noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Closing to fill small gaps in the curve
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Step 6: Remove small connected components (noise)
        binary = self._remove_small_components(binary, min_size=50)
        
        return FilterResult(
            filtered_image=cleaned,
            binary_mask=binary,
            grid_removed=grid_was_removed,
            noise_level=noise_level
        )
    
    def _estimate_noise(self, gray: np.ndarray) -> float:
        """
        Estimate the noise level in the image using Laplacian variance.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Noise level estimate (higher = more noise)
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-1 range (empirically determined thresholds)
        normalized = min(variance / 1000.0, 1.0)
        return float(normalized)
    
    def _remove_grid_lines(
        self,
        gray: np.ndarray
    ) -> Tuple[np.ndarray, bool]:
        """
        Remove horizontal and vertical grid lines.
        
        Uses morphological operations with directional kernels
        to detect and remove grid lines while preserving the curve.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Tuple of (processed image, whether grid was detected/removed)
        """
        # Create horizontal kernel
        horizontal_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.grid_kernel_size, 1)
        )
        
        # Create vertical kernel
        vertical_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (1, self.grid_kernel_size)
        )
        
        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(
            gray, cv2.MORPH_OPEN, horizontal_kernel, iterations=2
        )
        
        # Detect vertical lines
        vertical_lines = cv2.morphologyEx(
            gray, cv2.MORPH_OPEN, vertical_kernel, iterations=2
        )
        
        # Combine detected lines
        grid_mask = cv2.add(horizontal_lines, vertical_lines)
        
        # Check if significant grid was detected
        grid_pixels = np.sum(grid_mask > 127)
        total_pixels = gray.shape[0] * gray.shape[1]
        grid_ratio = grid_pixels / total_pixels
        
        grid_detected = grid_ratio > 0.01  # More than 1% of image is grid
        
        if grid_detected:
            # Remove grid lines from original image
            # Use inpainting for smooth removal
            _, grid_binary = cv2.threshold(grid_mask, 127, 255, cv2.THRESH_BINARY)
            
            # Dilate the mask slightly to ensure complete removal
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            grid_binary = cv2.dilate(grid_binary, dilate_kernel, iterations=1)
            
            # Inpaint to fill grid areas
            result = cv2.inpaint(gray, grid_binary, 3, cv2.INPAINT_TELEA)
            
            logger.debug(f"Grid lines removed (ratio: {grid_ratio:.4f})")
            return result, True
        
        return gray, False
    
    def _remove_watermarks(self, gray: np.ndarray) -> np.ndarray:
        """
        Attempt to remove watermarks using contrast enhancement.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Image with reduced watermark visibility
        """
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Detect potential watermark regions (low contrast areas)
        local_mean = cv2.blur(gray, (50, 50))
        local_std = cv2.blur(np.abs(gray.astype(float) - local_mean.astype(float)), (50, 50))
        
        # Areas with low local standard deviation might be watermarks
        watermark_mask = (local_std < 10).astype(np.uint8) * 255
        
        # If significant watermark detected, apply additional filtering
        if np.sum(watermark_mask) > 0.05 * gray.shape[0] * gray.shape[1]:
            # Use bilateral filter to smooth while preserving edges
            result = cv2.bilateralFilter(enhanced, 9, 75, 75)
            logger.debug("Watermark filtering applied")
            return result
        
        return enhanced
    
    def _remove_small_components(
        self,
        binary: np.ndarray,
        min_size: int = 50
    ) -> np.ndarray:
        """
        Remove small connected components (noise) from binary image.
        
        Args:
            binary: Binary image
            min_size: Minimum component size to keep
            
        Returns:
            Cleaned binary image
        """
        # Find connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        # Create output image
        result = np.zeros_like(binary)
        
        # Keep only components larger than min_size
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                result[labels == i] = 255
        
        return result
    
    def isolate_curve(
        self,
        binary: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Isolate the main UCS curve from the binary mask.
        
        Uses heuristics to identify the primary stress-strain curve:
        - Should span significant horizontal distance
        - Should start from lower-left region
        - Should be continuous or nearly continuous
        
        Args:
            binary: Binary mask from filtering
            image_shape: Original image shape (height, width)
            
        Returns:
            Binary mask containing only the main curve
        """
        height, width = image_shape
        
        # Find all contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return binary
        
        # Score each contour based on UCS curve characteristics
        scored_contours = []
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            
            # Calculate scores
            width_score = w / width  # Should span significant width
            
            # Should be in the upper portion (curves go up)
            position_score = 1.0 - (y / height)
            
            # Should have reasonable aspect ratio (wider than tall typically)
            aspect_score = min(w / max(h, 1), 3) / 3
            
            # Combine scores
            total_score = (
                0.4 * width_score +
                0.3 * position_score +
                0.3 * aspect_score
            )
            
            scored_contours.append((total_score, contour))
        
        # Keep the highest scoring contour(s)
        scored_contours.sort(key=lambda x: x[0], reverse=True)
        
        result = np.zeros_like(binary)
        
        # Draw the best contour(s)
        if scored_contours:
            best_score = scored_contours[0][0]
            for score, contour in scored_contours:
                # Keep contours within 50% of best score
                if score >= best_score * 0.5:
                    cv2.drawContours(result, [contour], -1, 255, -1)
        
        return result
