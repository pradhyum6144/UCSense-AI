"""
Peak Detection Module

Automatically identifies Peak UCS value and Failure Strain by analyzing
the stress-strain curve's derivative to locate the global maximum.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import interp1d
import logging

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class UCSFeatures:
    """Extracted UCS test features."""
    peak_stress: float  # Peak UCS value (kN/m²)
    failure_strain: float  # Strain at peak stress (%)
    initial_modulus: float  # Initial tangent modulus (kN/m²/%)
    secant_modulus_50: float  # Secant modulus at 50% peak stress
    energy_to_peak: float  # Area under curve up to peak
    post_peak_detected: bool  # Whether post-peak behavior was captured
    confidence: float  # Confidence in the detected values


@dataclass
class CurveAnalysis:
    """Complete curve analysis result."""
    features: UCSFeatures
    strain_data: np.ndarray
    stress_data: np.ndarray
    smoothed_stress: np.ndarray
    derivative: np.ndarray
    peak_index: int


class PeakDetector:
    """
    Detects UCS curve features through derivative analysis.
    
    Uses Savitzky-Golay filtering for smoothing and calculates
    the stress derivative to accurately locate the peak.
    """
    
    def __init__(
        self,
        window_length: int = None,
        polyorder: int = 3,
        min_peak_prominence: float = 0.1
    ):
        """
        Initialize the peak detector.
        
        Args:
            window_length: Savitzky-Golay window length (odd number)
            polyorder: Polynomial order for Savitzky-Golay filter
            min_peak_prominence: Minimum prominence for peak detection
        """
        self.window_length = window_length or settings.curve_smoothness_window
        self.polyorder = polyorder
        self.min_peak_prominence = min_peak_prominence
    
    def analyze(
        self,
        strain: List[float],
        stress: List[float]
    ) -> CurveAnalysis:
        """
        Analyze the stress-strain curve to extract features.
        
        Args:
            strain: Strain values (%)
            stress: Stress values (kN/m²)
            
        Returns:
            CurveAnalysis with all extracted features
        """
        # Convert to numpy arrays
        strain_arr = np.array(strain)
        stress_arr = np.array(stress)
        
        # Ensure ascending strain order
        sort_idx = np.argsort(strain_arr)
        strain_arr = strain_arr[sort_idx]
        stress_arr = stress_arr[sort_idx]
        
        # Remove duplicate strain values (keep average stress)
        strain_arr, unique_idx = np.unique(strain_arr, return_index=True)
        stress_arr = stress_arr[unique_idx]
        
        # Ensure minimum data points
        if len(strain_arr) < 5:
            logger.warning("Insufficient data points for analysis")
            return self._create_empty_result(strain_arr, stress_arr)
        
        # Interpolate to uniform spacing for better analysis
        strain_uniform, stress_uniform = self._interpolate_uniform(
            strain_arr, stress_arr, num_points=200
        )
        
        # Apply Savitzky-Golay smoothing
        window = min(self.window_length, len(stress_uniform) - 1)
        if window % 2 == 0:
            window -= 1
        window = max(5, window)
        
        smoothed = savgol_filter(stress_uniform, window, self.polyorder)
        
        # Calculate derivative (dσ/dε)
        derivative = np.gradient(smoothed, strain_uniform)
        
        # Find the peak
        peak_index = self._find_peak(smoothed, derivative)
        
        # Extract features
        features = self._extract_features(
            strain_uniform, stress_uniform, smoothed, derivative, peak_index
        )
        
        return CurveAnalysis(
            features=features,
            strain_data=strain_uniform,
            stress_data=stress_uniform,
            smoothed_stress=smoothed,
            derivative=derivative,
            peak_index=peak_index
        )
    
    def _interpolate_uniform(
        self,
        strain: np.ndarray,
        stress: np.ndarray,
        num_points: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate data to uniform strain spacing.
        
        Args:
            strain: Original strain values
            stress: Original stress values
            num_points: Number of output points
            
        Returns:
            Tuple of (uniform_strain, interpolated_stress)
        """
        # Create interpolation function
        interp_func = interp1d(
            strain, stress, 
            kind='linear', 
            fill_value='extrapolate'
        )
        
        # Generate uniform strain values
        strain_uniform = np.linspace(
            strain.min(), strain.max(), num_points
        )
        
        # Interpolate stress
        stress_uniform = interp_func(strain_uniform)
        
        return strain_uniform, stress_uniform
    
    def _find_peak(
        self,
        stress: np.ndarray,
        derivative: np.ndarray
    ) -> int:
        """
        Find the peak stress location.
        
        Uses multiple methods:
        1. Zero-crossing of derivative (most accurate)
        2. Peak finding with prominence
        3. Global maximum (fallback)
        
        Args:
            stress: Smoothed stress values
            derivative: Stress derivative
            
        Returns:
            Index of peak stress
        """
        # Method 1: Find zero-crossings of derivative (going from + to -)
        zero_crossings = []
        for i in range(1, len(derivative)):
            if derivative[i-1] > 0 and derivative[i] <= 0:
                # Interpolate for more precise location
                if derivative[i-1] != derivative[i]:
                    frac = derivative[i-1] / (derivative[i-1] - derivative[i])
                    zero_crossings.append(i - 1 + frac)
                else:
                    zero_crossings.append(i)
        
        if zero_crossings:
            # Take the first significant zero-crossing (the peak)
            for zc in zero_crossings:
                idx = int(round(zc))
                # Verify it's actually a peak (stress is high)
                if stress[idx] > 0.5 * stress.max():
                    logger.debug(f"Peak found via zero-crossing at index {idx}")
                    return idx
        
        # Method 2: Use scipy peak finding
        peaks, properties = find_peaks(
            stress,
            prominence=self.min_peak_prominence * stress.max()
        )
        
        if len(peaks) > 0:
            # Select peak with highest stress
            best_peak = peaks[np.argmax(stress[peaks])]
            logger.debug(f"Peak found via peak finding at index {best_peak}")
            return int(best_peak)
        
        # Method 3: Fallback to global maximum
        peak_idx = int(np.argmax(stress))
        logger.debug(f"Peak found via global maximum at index {peak_idx}")
        return peak_idx
    
    def _extract_features(
        self,
        strain: np.ndarray,
        stress: np.ndarray,
        smoothed: np.ndarray,
        derivative: np.ndarray,
        peak_index: int
    ) -> UCSFeatures:
        """
        Extract all UCS features from the curve.
        
        Args:
            strain: Strain values
            stress: Original stress values
            smoothed: Smoothed stress values
            derivative: Stress derivative
            peak_index: Index of peak stress
            
        Returns:
            UCSFeatures dataclass
        """
        # Peak stress and failure strain
        peak_stress = float(smoothed[peak_index])
        failure_strain = float(strain[peak_index])
        
        # Initial tangent modulus (slope at the beginning)
        # Use first 10% of data
        initial_range = max(1, int(0.1 * peak_index))
        initial_modulus = float(np.mean(derivative[:initial_range]))
        
        # Secant modulus at 50% peak stress
        half_peak = peak_stress / 2
        idx_50 = np.argmin(np.abs(smoothed[:peak_index+1] - half_peak))
        if strain[idx_50] > 0:
            secant_modulus_50 = float(smoothed[idx_50] / strain[idx_50])
        else:
            secant_modulus_50 = initial_modulus
        
        # Energy to peak (area under curve)
        energy_to_peak = float(np.trapz(smoothed[:peak_index+1], strain[:peak_index+1]))
        
        # Check for post-peak behavior
        post_peak_detected = peak_index < len(strain) - 5
        
        # Calculate confidence based on curve quality
        confidence = self._calculate_confidence(
            strain, smoothed, derivative, peak_index
        )
        
        return UCSFeatures(
            peak_stress=peak_stress,
            failure_strain=failure_strain,
            initial_modulus=initial_modulus,
            secant_modulus_50=secant_modulus_50,
            energy_to_peak=energy_to_peak,
            post_peak_detected=post_peak_detected,
            confidence=confidence
        )
    
    def _calculate_confidence(
        self,
        strain: np.ndarray,
        stress: np.ndarray,
        derivative: np.ndarray,
        peak_index: int
    ) -> float:
        """
        Calculate confidence score for the detected features.
        
        Args:
            strain: Strain values
            stress: Stress values
            derivative: Derivative values
            peak_index: Detected peak index
            
        Returns:
            Confidence score (0-1)
        """
        scores = []
        
        # 1. Data density score
        if len(strain) >= 100:
            scores.append(1.0)
        else:
            scores.append(len(strain) / 100)
        
        # 2. Peak prominence score
        peak_stress = stress[peak_index]
        if peak_stress > 0:
            min_stress = stress.min()
            prominence = (peak_stress - min_stress) / peak_stress
            scores.append(min(prominence, 1.0))
        else:
            scores.append(0.0)
        
        # 3. Derivative behavior score (should be positive before peak, negative after)
        if peak_index > 5 and peak_index < len(derivative) - 5:
            pre_peak_deriv = derivative[:peak_index]
            post_peak_deriv = derivative[peak_index:]
            
            pre_positive = np.mean(pre_peak_deriv > 0)
            post_negative = np.mean(post_peak_deriv < 0) if len(post_peak_deriv) > 0 else 0.5
            
            scores.append((pre_positive + post_negative) / 2)
        else:
            scores.append(0.5)
        
        # 4. Monotonicity before peak (strain should increase, stress should mostly increase)
        pre_peak_strain = strain[:peak_index+1]
        strain_increases = np.sum(np.diff(pre_peak_strain) > 0) / max(len(pre_peak_strain)-1, 1)
        scores.append(strain_increases)
        
        return float(np.mean(scores))
    
    def _create_empty_result(
        self,
        strain: np.ndarray,
        stress: np.ndarray
    ) -> CurveAnalysis:
        """
        Create an empty result for insufficient data.
        
        Args:
            strain: Strain array
            stress: Stress array
            
        Returns:
            CurveAnalysis with default values
        """
        features = UCSFeatures(
            peak_stress=float(stress.max()) if len(stress) > 0 else 0.0,
            failure_strain=float(strain[np.argmax(stress)]) if len(stress) > 0 else 0.0,
            initial_modulus=0.0,
            secant_modulus_50=0.0,
            energy_to_peak=0.0,
            post_peak_detected=False,
            confidence=0.0
        )
        
        return CurveAnalysis(
            features=features,
            strain_data=strain,
            stress_data=stress,
            smoothed_stress=stress,
            derivative=np.zeros_like(stress),
            peak_index=int(np.argmax(stress)) if len(stress) > 0 else 0
        )
    
    def get_critical_points(
        self,
        analysis: CurveAnalysis
    ) -> List[Tuple[float, float, str]]:
        """
        Get critical points on the curve for annotation.
        
        Args:
            analysis: Complete curve analysis
            
        Returns:
            List of (strain, stress, label) tuples
        """
        points = []
        
        # Peak point
        points.append((
            analysis.features.failure_strain,
            analysis.features.peak_stress,
            f"Peak UCS: {analysis.features.peak_stress:.1f} kN/m²"
        ))
        
        # 50% peak point
        half_peak = analysis.features.peak_stress / 2
        idx_50 = np.argmin(np.abs(
            analysis.smoothed_stress[:analysis.peak_index+1] - half_peak
        ))
        points.append((
            float(analysis.strain_data[idx_50]),
            float(analysis.smoothed_stress[idx_50]),
            "50% Peak"
        ))
        
        # Initial linear region end (where derivative starts decreasing significantly)
        initial_deriv = analysis.derivative[:max(1, analysis.peak_index // 2)]
        if len(initial_deriv) > 5:
            deriv_change = np.diff(initial_deriv)
            # Find where derivative starts decreasing
            decreasing_idx = np.where(deriv_change < -0.1 * initial_deriv.max())[0]
            if len(decreasing_idx) > 0:
                linear_end = decreasing_idx[0]
                points.append((
                    float(analysis.strain_data[linear_end]),
                    float(analysis.smoothed_stress[linear_end]),
                    "Yield Point"
                ))
        
        return points
