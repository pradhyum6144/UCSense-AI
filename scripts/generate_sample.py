"""
Generate a sample UCS graph image for testing the extraction pipeline.
"""

import cv2
import numpy as np
import os

def create_sample_ucs_graph():
    """Create a realistic-looking UCS stress-strain graph."""
    
    # Create white background (800x600)
    width, height = 800, 600
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Define plot area
    margin_left = 100
    margin_right = 50
    margin_top = 50
    margin_bottom = 80
    
    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = height - margin_bottom
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top
    
    # Draw axes
    cv2.line(img, (plot_left, plot_bottom), (plot_right, plot_bottom), (0, 0, 0), 2)  # X-axis
    cv2.line(img, (plot_left, plot_bottom), (plot_left, plot_top), (0, 0, 0), 2)  # Y-axis
    
    # Add arrow heads
    cv2.arrowedLine(img, (plot_right - 20, plot_bottom), (plot_right, plot_bottom), (0, 0, 0), 2)
    cv2.arrowedLine(img, (plot_left, plot_top + 20), (plot_left, plot_top), (0, 0, 0), 2)
    
    # Grid lines (light gray)
    grid_color = (200, 200, 200)
    for i in range(1, 6):
        x = plot_left + int(i * plot_width / 6)
        cv2.line(img, (x, plot_top), (x, plot_bottom), grid_color, 1)
    for i in range(1, 5):
        y = plot_bottom - int(i * plot_height / 5)
        cv2.line(img, (plot_left, y), (plot_right, y), grid_color, 1)
    
    # X-axis tick marks and labels (Strain %)
    x_values = [0, 1, 2, 3, 4, 5, 6]
    for i, val in enumerate(x_values):
        x = plot_left + int(i * plot_width / 6)
        cv2.line(img, (x, plot_bottom), (x, plot_bottom + 10), (0, 0, 0), 2)
        cv2.putText(img, str(val), (x - 5, plot_bottom + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Y-axis tick marks and labels (Stress kN/m²)
    y_values = [0, 1000, 2000, 3000, 4000, 5000]
    for i, val in enumerate(y_values):
        y = plot_bottom - int(i * plot_height / 5)
        cv2.line(img, (plot_left - 10, y), (plot_left, y), (0, 0, 0), 2)
        cv2.putText(img, str(val), (plot_left - 60, y + 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Axis labels
    cv2.putText(img, "Strain (%)", (width // 2 - 40, height - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Rotated Y-axis label (simulate with horizontal text)
    cv2.putText(img, "Stress", (15, height // 2 - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, "(kN/m2)", (10, height // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Generate UCS curve data
    # Typical UCS curve: rises rapidly, peaks around 2-3% strain, then drops
    strain_data = np.linspace(0, 5.5, 200)
    
    # UCS curve formula: stress = a * strain * exp(-b * strain) + noise
    peak_strain = 2.5  # Strain at peak
    peak_stress = 4200  # Peak stress
    
    # Calculate stress values
    stress_data = []
    for strain in strain_data:
        # Modified exponential curve
        if strain <= peak_strain:
            # Rising portion
            stress = peak_stress * (strain / peak_strain) ** 0.8
        else:
            # Falling portion
            decay = (strain - peak_strain) / (5.5 - peak_strain)
            stress = peak_stress * (1 - 0.4 * decay ** 1.5)
        
        # Add slight noise
        noise = np.random.normal(0, 30)
        stress_data.append(max(0, stress + noise))
    
    stress_data = np.array(stress_data)
    
    # Convert to pixel coordinates
    def strain_to_x(strain):
        return int(plot_left + (strain / 6) * plot_width)
    
    def stress_to_y(stress):
        return int(plot_bottom - (stress / 5000) * plot_height)
    
    # Draw the curve
    points = []
    for strain, stress in zip(strain_data, stress_data):
        x = strain_to_x(strain)
        y = stress_to_y(stress)
        points.append((x, y))
    
    # Draw curve with thickness
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i + 1], (0, 0, 180), 2)  # Dark blue curve
    
    # Mark the peak point
    peak_idx = np.argmax(stress_data)
    peak_x = strain_to_x(strain_data[peak_idx])
    peak_y = stress_to_y(stress_data[peak_idx])
    cv2.circle(img, (peak_x, peak_y), 5, (0, 0, 255), -1)  # Red marker
    
    # Add title
    cv2.putText(img, "UCS Test - Sample A1", (width // 2 - 100, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Save the image
    output_dir = "/home/pradhyum/L&T/data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "sample_ucs_graph.png")
    cv2.imwrite(output_path, img)
    
    print(f"Sample UCS graph saved to: {output_path}")
    print(f"Peak stress: {stress_data[peak_idx]:.1f} kN/m² at {strain_data[peak_idx]:.2f}% strain")
    
    return output_path, strain_data[peak_idx], stress_data[peak_idx]


if __name__ == "__main__":
    create_sample_ucs_graph()
