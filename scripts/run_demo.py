"""
Standalone Demo Script for UCSense-AI Extraction Pipeline

This script demonstrates the core extraction capabilities without 
requiring Tesseract OCR (which needs system installation).
"""

import sys
sys.path.insert(0, '/home/pradhyum/L&T/backend')

import cv2
import numpy as np
import json
from pathlib import Path

# Import our modules
from preprocessing.rectifier import ImageRectifier
from preprocessing.filter import MorphologicalFilter
from extraction.contour_tracer import ContourTracer
from analysis.peak_detector import PeakDetector

def run_demo():
    """Run the extraction pipeline demo."""
    
    print("=" * 60)
    print("  UCSense-AI Extraction Pipeline Demo")
    print("=" * 60)
    
    # Load sample image
    image_path = "/home/pradhyum/L&T/data/sample_ucs_graph.png"
    print(f"\n📷 Loading image: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Failed to load image!")
        return
    
    print(f"   Image size: {image.shape[1]}x{image.shape[0]} pixels")
    
    # Step 1: Image Rectification
    print("\n🔧 Step 1: Image Rectification")
    rectifier = ImageRectifier()
    rect_result = rectifier.rectify(image)
    print(f"   Rotation angle: {rect_result.rotation_angle:.2f}°")
    print(f"   Corrected: {rect_result.was_corrected}")
    
    # Step 2: Morphological Filtering
    print("\n🔧 Step 2: Morphological Filtering")
    filter_module = MorphologicalFilter()
    filter_result = filter_module.filter(rect_result.image)
    print(f"   Noise level: {filter_result.noise_level:.4f}")
    print(f"   Grid removed: {filter_result.grid_removed}")
    
    # Step 3: Curve Extraction
    print("\n🔧 Step 3: Curve Extraction (Contour Tracing)")
    tracer = ContourTracer()
    
    # Define plot region (we know the image structure)
    # In real use, this comes from axis detection
    plot_region = (100, 50, 650, 420)  # x, y, w, h
    
    extraction_result = tracer.extract(filter_result.binary_mask, plot_region)
    print(f"   Points extracted: {len(extraction_result.points)}")
    print(f"   Smoothed points: {len(extraction_result.smoothed_points)}")
    print(f"   Extraction confidence: {extraction_result.confidence:.2f}")
    
    # Step 4: Convert to Engineering Units (Manual Scale for Demo)
    # We know the sample image has:
    # - X-axis: 0-6% strain
    # - Y-axis: 0-5000 kN/m²
    # - Plot region: x=100-750, y=50-520
    
    print("\n🔧 Step 4: Coordinate Mapping")
    
    x_min_px, y_max_px = 100, 520  # Origin in pixels
    x_max_px, y_min_px = 750, 50   # Opposite corner
    
    strain_min, strain_max = 0, 6
    stress_min, stress_max = 0, 5000
    
    # Scale factors
    strain_per_px = (strain_max - strain_min) / (x_max_px - x_min_px)
    stress_per_px = (stress_max - stress_min) / (y_max_px - y_min_px)
    
    print(f"   Strain scale: {strain_per_px:.4f} %/pixel")
    print(f"   Stress scale: {stress_per_px:.2f} kN/m²/pixel")
    
    # Map points to engineering units
    data_points = []
    for px, py in extraction_result.smoothed_points:
        if x_min_px <= px <= x_max_px and y_min_px <= py <= y_max_px:
            strain = (px - x_min_px) * strain_per_px
            stress = (y_max_px - py) * stress_per_px
            data_points.append({
                'strain': round(strain, 4),
                'stress': round(stress, 2)
            })
    
    print(f"   Mapped data points: {len(data_points)}")
    
    # Step 5: Feature Analysis
    print("\n🔧 Step 5: Peak Detection & Feature Analysis")
    
    if data_points:
        strains = [p['strain'] for p in data_points]
        stresses = [p['stress'] for p in data_points]
        
        peak_detector = PeakDetector()
        analysis = peak_detector.analyze(strains, stresses)
        
        features = analysis.features
        
        print(f"\n📊 EXTRACTED FEATURES:")
        print("   " + "-" * 40)
        print(f"   Peak UCS:         {features.peak_stress:.1f} kN/m²")
        print(f"   Failure Strain:   {features.failure_strain:.3f} %")
        print(f"   Initial Modulus:  {features.initial_modulus:.0f} kN/m²/%")
        print(f"   Secant Modulus:   {features.secant_modulus_50:.0f} kN/m²/%")
        print(f"   Energy to Peak:   {features.energy_to_peak:.2f} kJ/m³")
        print(f"   Analysis Confidence: {features.confidence:.2f}")
        print("   " + "-" * 40)
        
        # Expected values from our sample generation
        print(f"\n✅ EXPECTED VALUES (Ground Truth):")
        print(f"   Peak Stress: ~4252 kN/m² at ~2.54% strain")
        
        # Calculate error
        expected_peak = 4252
        peak_error = abs(features.peak_stress - expected_peak) / expected_peak * 100
        print(f"\n📈 Peak Detection Error: {peak_error:.1f}%")
        
        if peak_error < 10:
            print("   ✅ Excellent accuracy!")
        elif peak_error < 20:
            print("   ⚠️ Good accuracy, tune parameters for improvement")
        else:
            print("   ❌ High error - OCR scale mapping needed")
    
    # Save visualization
    print("\n🖼️ Creating visualization...")
    
    output_img = rect_result.image.copy()
    
    # Draw extracted curve in green
    for i in range(len(extraction_result.smoothed_points) - 1):
        pt1 = extraction_result.smoothed_points[i]
        pt2 = extraction_result.smoothed_points[i + 1]
        cv2.line(output_img, pt1, pt2, (0, 255, 0), 2)
    
    # Mark peak point
    if data_points:
        peak_idx = analysis.peak_index
        if 0 <= peak_idx < len(extraction_result.smoothed_points):
            peak_pt = extraction_result.smoothed_points[peak_idx]
            cv2.circle(output_img, peak_pt, 10, (0, 0, 255), 3)
            cv2.putText(output_img, f"Peak: {features.peak_stress:.0f} kN/m2", 
                       (peak_pt[0] + 15, peak_pt[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    output_path = "/home/pradhyum/L&T/data/extraction_result.png"
    cv2.imwrite(output_path, output_img)
    print(f"   Saved: {output_path}")
    
    # Save data as JSON
    result_data = {
        "data_points": data_points[:10],  # First 10 points
        "total_points": len(data_points),
        "features": {
            "peak_stress_kN_m2": features.peak_stress,
            "failure_strain_pct": features.failure_strain,
            "initial_modulus": features.initial_modulus,
            "confidence": features.confidence
        }
    }
    
    json_path = "/home/pradhyum/L&T/data/extraction_result.json"
    with open(json_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"   Saved: {json_path}")
    
    print("\n" + "=" * 60)
    print("  Demo Complete!")
    print("=" * 60)
    
    return result_data


if __name__ == "__main__":
    run_demo()
