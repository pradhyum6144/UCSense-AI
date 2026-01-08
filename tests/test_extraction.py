"""
Test Suite for UCSense-AI Backend

Comprehensive tests for the extraction pipeline.
"""

import pytest
import numpy as np
from pathlib import Path
import cv2

# Test fixtures
@pytest.fixture
def sample_image():
    """Create a synthetic UCS graph image for testing."""
    # Create a 800x600 white image
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw axes
    cv2.line(img, (100, 500), (700, 500), (0, 0, 0), 2)  # X-axis
    cv2.line(img, (100, 500), (100, 100), (0, 0, 0), 2)  # Y-axis
    
    # Draw a synthetic UCS curve
    points = []
    for x in range(100, 600, 5):
        # Simulate stress-strain curve: rises then falls
        t = (x - 100) / 500
        stress = 400 * (1 - (t - 0.7)**2) if t < 0.7 else 400 * (1 - (t - 0.7)**2) * 0.9
        y = int(500 - stress)
        points.append((x, y))
    
    for i in range(len(points) - 1):
        cv2.line(img, points[i], points[i+1], (0, 0, 0), 2)
    
    # Add tick marks and labels
    for i, value in enumerate([0, 2, 4, 6, 8]):
        x = 100 + i * 125
        cv2.line(img, (x, 500), (x, 510), (0, 0, 0), 1)
        cv2.putText(img, str(value), (x-10, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    for i, value in enumerate([0, 100, 200, 300, 400]):
        y = 500 - i * 100
        cv2.line(img, (90, y), (100, y), (0, 0, 0), 1)
        cv2.putText(img, str(value), (50, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return img


@pytest.fixture
def noisy_image(sample_image):
    """Add noise to the sample image."""
    noise = np.random.randint(0, 30, sample_image.shape, dtype=np.uint8)
    noisy = cv2.add(sample_image, noise)
    return noisy


class TestImageRectifier:
    """Tests for the ImageRectifier class."""
    
    def test_rectify_straight_image(self, sample_image):
        """Test that a straight image passes through unchanged."""
        from preprocessing.rectifier import ImageRectifier
        
        rectifier = ImageRectifier()
        result = rectifier.rectify(sample_image)
        
        assert result.image is not None
        assert result.image.shape[0] > 0
        assert result.image.shape[1] > 0
        # Straight image should have minimal rotation
        assert abs(result.rotation_angle) < 5
    
    def test_rectify_skewed_image(self, sample_image):
        """Test rectification of a skewed image."""
        from preprocessing.rectifier import ImageRectifier
        
        # Rotate image by 5 degrees
        h, w = sample_image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, 5, 1.0)
        skewed = cv2.warpAffine(sample_image, matrix, (w, h))
        
        rectifier = ImageRectifier()
        result = rectifier.rectify(skewed)
        
        assert result.was_corrected or abs(result.rotation_angle) < 1


class TestMorphologicalFilter:
    """Tests for the MorphologicalFilter class."""
    
    def test_filter_clean_image(self, sample_image):
        """Test filtering on a clean image."""
        from preprocessing.filter import MorphologicalFilter
        
        filter_module = MorphologicalFilter()
        result = filter_module.filter(sample_image)
        
        assert result.filtered_image is not None
        assert result.binary_mask is not None
        assert 0 <= result.noise_level <= 1
    
    def test_filter_noisy_image(self, noisy_image):
        """Test filtering on a noisy image."""
        from preprocessing.filter import MorphologicalFilter
        
        filter_module = MorphologicalFilter()
        result = filter_module.filter(noisy_image)
        
        assert result.filtered_image is not None
        # Noisy image should have higher noise level
        assert result.noise_level > 0


class TestAxisDetector:
    """Tests for the AxisDetector class."""
    
    def test_detect_axes(self, sample_image):
        """Test axis detection on a synthetic image."""
        from ocr.axis_detector import AxisDetector
        
        detector = AxisDetector()
        result = detector.detect(sample_image)
        
        assert result.origin_pixel is not None
        assert result.plot_region is not None
        # Should detect some tick marks
        # Note: OCR may not work perfectly on synthetic images


class TestScaleMapper:
    """Tests for the ScaleMapper class."""
    
    def test_calculate_scale_with_valid_ticks(self):
        """Test scale calculation with valid tick marks."""
        from ocr.scale_mapper import ScaleMapper
        from ocr.axis_detector import AxisInfo, TickMark
        
        # Create mock tick marks
        x_axis = AxisInfo(orientation="horizontal")
        x_axis.tick_marks = [
            TickMark(pixel_position=100, value=0, confidence=0.9, raw_text="0"),
            TickMark(pixel_position=200, value=2, confidence=0.9, raw_text="2"),
            TickMark(pixel_position=300, value=4, confidence=0.9, raw_text="4"),
        ]
        x_axis.is_valid = True
        
        y_axis = AxisInfo(orientation="vertical")
        y_axis.tick_marks = [
            TickMark(pixel_position=500, value=0, confidence=0.9, raw_text="0"),
            TickMark(pixel_position=400, value=100, confidence=0.9, raw_text="100"),
            TickMark(pixel_position=300, value=200, confidence=0.9, raw_text="200"),
        ]
        y_axis.is_valid = True
        
        mapper = ScaleMapper()
        scale = mapper.calculate_scale_factors(x_axis, y_axis, (100, 500))
        
        assert scale.is_valid
        assert scale.x_r_squared > 0.9
        assert scale.y_r_squared > 0.9


class TestContourTracer:
    """Tests for the ContourTracer class."""
    
    def test_extract_from_binary_mask(self, sample_image):
        """Test curve extraction from binary mask."""
        from extraction.contour_tracer import ContourTracer
        
        # Create binary mask from image
        gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        tracer = ContourTracer()
        result = tracer.extract(binary)
        
        assert len(result.points) > 0
        assert result.confidence >= 0


class TestPeakDetector:
    """Tests for the PeakDetector class."""
    
    def test_detect_peak_in_synthetic_data(self):
        """Test peak detection on synthetic stress-strain data."""
        from analysis.peak_detector import PeakDetector
        
        # Create synthetic data with clear peak
        strain = np.linspace(0, 10, 100)
        stress = 1000 * strain * np.exp(-strain / 3)  # Peak around strain=3
        
        detector = PeakDetector()
        analysis = detector.analyze(strain.tolist(), stress.tolist())
        
        assert analysis.features.peak_stress > 0
        assert analysis.features.failure_strain > 0
        assert 2 < analysis.features.failure_strain < 4  # Expected peak region
        assert analysis.features.confidence > 0.5


class TestConfidenceScorer:
    """Tests for the ConfidenceScorer class."""
    
    def test_calculate_high_confidence(self):
        """Test scoring with high-quality inputs."""
        from analysis.confidence_scorer import ConfidenceScorer
        
        scorer = ConfidenceScorer()
        
        # High-quality mock data
        ocr_confidences = [0.9, 0.85, 0.92, 0.88]
        curve_points = [(i * 0.1, i * 100) for i in range(50)]  # Smooth curve
        axis_tick_counts = (5, 5)
        image_quality = 0.85
        
        report = scorer.calculate_score(
            ocr_confidences=ocr_confidences,
            curve_points=curve_points,
            axis_tick_counts=axis_tick_counts,
            image_quality_score=image_quality,
            extraction_method="contour_tracing"
        )
        
        assert report.overall_score > 0.7
        assert report.grade in ['A', 'B', 'C']
    
    def test_calculate_low_confidence(self):
        """Test scoring with low-quality inputs."""
        from analysis.confidence_scorer import ConfidenceScorer
        
        scorer = ConfidenceScorer()
        
        # Low-quality mock data
        ocr_confidences = [0.3, 0.4]
        curve_points = [(i, i * 10 + np.random.randint(-50, 50)) for i in range(10)]
        axis_tick_counts = (1, 1)
        image_quality = 0.3
        
        report = scorer.calculate_score(
            ocr_confidences=ocr_confidences,
            curve_points=curve_points,
            axis_tick_counts=axis_tick_counts,
            image_quality_score=image_quality,
            extraction_method="extraction_failed"
        )
        
        assert report.overall_score < 0.7
        assert len(report.warnings) > 0


class TestAPIEndpoints:
    """Tests for FastAPI endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from main import app
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_extract_without_file(self, client):
        """Test extraction endpoint without file."""
        response = client.post("/api/v1/extract")
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_extract_with_invalid_file(self, client):
        """Test extraction endpoint with invalid file type."""
        response = client.post(
            "/api/v1/extract",
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
