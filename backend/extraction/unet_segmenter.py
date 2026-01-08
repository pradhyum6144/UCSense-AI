"""
U-Net Segmenter Module

Deep learning-based curve segmentation using U-Net architecture
for handling noisy or degraded graph images.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import logging
import cv2

logger = logging.getLogger(__name__)

# TensorFlow imports with graceful fallback
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available, U-Net segmentation disabled")


@dataclass
class SegmentationResult:
    """Result of U-Net segmentation."""
    mask: np.ndarray  # Binary segmentation mask
    probability_map: np.ndarray  # Probability values
    confidence: float  # Average probability of detected curve
    points: List[Tuple[int, int]]  # Extracted curve points


class UNetSegmenter:
    """
    U-Net based curve segmentation for noisy images.
    
    Uses a pre-trained or custom U-Net model to segment the UCS curve
    from challenging images with noise, watermarks, or degradation.
    """
    
    DEFAULT_INPUT_SIZE = (256, 256)
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
        threshold: float = 0.5
    ):
        """
        Initialize the U-Net segmenter.
        
        Args:
            model_path: Path to pre-trained model weights
            input_size: Expected input size for the model
            threshold: Probability threshold for binary mask
        """
        self.model_path = model_path
        self.input_size = input_size
        self.threshold = threshold
        self.model = None
        self._model_loaded = False
    
    def load_model(self) -> bool:
        """
        Load the U-Net model from disk.
        
        Returns:
            True if model loaded successfully
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available")
            return False
        
        if self.model_path and Path(self.model_path).exists():
            try:
                self.model = keras.models.load_model(self.model_path)
                self._model_loaded = True
                logger.info(f"Loaded U-Net model from {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
        
        # Build default model if no pre-trained available
        logger.info("Building default U-Net model")
        self.model = self._build_unet()
        self._model_loaded = True
        return True
    
    def _build_unet(self) -> 'keras.Model':
        """
        Build a U-Net model for curve segmentation.
        
        Architecture:
        - Encoder: 4 conv blocks with max pooling
        - Bottleneck: 512 feature channels
        - Decoder: 4 upsampling blocks with skip connections
        
        Returns:
            Compiled Keras model
        """
        inputs = keras.Input(shape=(*self.input_size, 1))
        
        # Encoder
        # Block 1
        c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
        c1 = layers.BatchNormalization()(c1)
        c1 = layers.Conv2D(64, 3, activation='relu', padding='same')(c1)
        c1 = layers.BatchNormalization()(c1)
        p1 = layers.MaxPooling2D((2, 2))(c1)
        p1 = layers.Dropout(0.1)(p1)
        
        # Block 2
        c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
        c2 = layers.BatchNormalization()(c2)
        c2 = layers.Conv2D(128, 3, activation='relu', padding='same')(c2)
        c2 = layers.BatchNormalization()(c2)
        p2 = layers.MaxPooling2D((2, 2))(c2)
        p2 = layers.Dropout(0.1)(p2)
        
        # Block 3
        c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
        c3 = layers.BatchNormalization()(c3)
        c3 = layers.Conv2D(256, 3, activation='relu', padding='same')(c3)
        c3 = layers.BatchNormalization()(c3)
        p3 = layers.MaxPooling2D((2, 2))(c3)
        p3 = layers.Dropout(0.2)(p3)
        
        # Block 4
        c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(p3)
        c4 = layers.BatchNormalization()(c4)
        c4 = layers.Conv2D(512, 3, activation='relu', padding='same')(c4)
        c4 = layers.BatchNormalization()(c4)
        p4 = layers.MaxPooling2D((2, 2))(c4)
        p4 = layers.Dropout(0.2)(p4)
        
        # Bottleneck
        c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(p4)
        c5 = layers.BatchNormalization()(c5)
        c5 = layers.Conv2D(1024, 3, activation='relu', padding='same')(c5)
        c5 = layers.BatchNormalization()(c5)
        c5 = layers.Dropout(0.3)(c5)
        
        # Decoder
        # Block 6
        u6 = layers.Conv2DTranspose(512, 2, strides=2, padding='same')(c5)
        u6 = layers.concatenate([u6, c4])
        c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(u6)
        c6 = layers.BatchNormalization()(c6)
        c6 = layers.Conv2D(512, 3, activation='relu', padding='same')(c6)
        c6 = layers.BatchNormalization()(c6)
        c6 = layers.Dropout(0.2)(c6)
        
        # Block 7
        u7 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(c6)
        u7 = layers.concatenate([u7, c3])
        c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(u7)
        c7 = layers.BatchNormalization()(c7)
        c7 = layers.Conv2D(256, 3, activation='relu', padding='same')(c7)
        c7 = layers.BatchNormalization()(c7)
        c7 = layers.Dropout(0.2)(c7)
        
        # Block 8
        u8 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(c7)
        u8 = layers.concatenate([u8, c2])
        c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(u8)
        c8 = layers.BatchNormalization()(c8)
        c8 = layers.Conv2D(128, 3, activation='relu', padding='same')(c8)
        c8 = layers.BatchNormalization()(c8)
        c8 = layers.Dropout(0.1)(c8)
        
        # Block 9
        u9 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(c8)
        u9 = layers.concatenate([u9, c1])
        c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(u9)
        c9 = layers.BatchNormalization()(c9)
        c9 = layers.Conv2D(64, 3, activation='relu', padding='same')(c9)
        c9 = layers.BatchNormalization()(c9)
        
        # Output
        outputs = layers.Conv2D(1, 1, activation='sigmoid')(c9)
        
        model = keras.Model(inputs, outputs)
        
        # Compile with dice loss + binary cross entropy
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss=self._dice_bce_loss,
            metrics=['accuracy', self._dice_coefficient]
        )
        
        return model
    
    @staticmethod
    def _dice_coefficient(y_true, y_pred, smooth=1):
        """Calculate Dice coefficient metric."""
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (
            tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
        )
    
    @staticmethod
    def _dice_bce_loss(y_true, y_pred):
        """Combined Dice loss and Binary Cross Entropy."""
        bce = keras.losses.binary_crossentropy(y_true, y_pred)
        dice = 1 - UNetSegmenter._dice_coefficient(y_true, y_pred)
        return bce + dice
    
    def segment(self, image: np.ndarray) -> SegmentationResult:
        """
        Segment the UCS curve from an image.
        
        Args:
            image: Input BGR or grayscale image
            
        Returns:
            SegmentationResult with mask and extracted points
        """
        if not self._model_loaded:
            if not self.load_model():
                return SegmentationResult(
                    mask=np.zeros(image.shape[:2], dtype=np.uint8),
                    probability_map=np.zeros(image.shape[:2], dtype=np.float32),
                    confidence=0.0,
                    points=[]
                )
        
        original_shape = image.shape[:2]
        
        # Preprocess image
        processed = self._preprocess(image)
        
        # Run inference
        prediction = self.model.predict(processed, verbose=0)
        
        # Post-process
        probability_map = prediction[0, :, :, 0]
        
        # Resize back to original size
        probability_map = cv2.resize(
            probability_map,
            (original_shape[1], original_shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # Create binary mask
        mask = (probability_map > self.threshold).astype(np.uint8) * 255
        
        # Extract curve points from mask
        points = self._extract_points(mask)
        
        # Calculate confidence
        if points:
            # Average probability at curve locations
            confidence = float(np.mean([
                probability_map[y, x] for x, y in points
            ]))
        else:
            confidence = 0.0
        
        logger.info(
            f"U-Net segmentation: {len(points)} points, "
            f"confidence={confidence:.2f}"
        )
        
        return SegmentationResult(
            mask=mask,
            probability_map=probability_map,
            confidence=confidence,
            points=points
        )
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input BGR or grayscale image
            
        Returns:
            Preprocessed array ready for model
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Resize to model input size
        resized = cv2.resize(gray, self.input_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions
        return normalized.reshape(1, *self.input_size, 1)
    
    def _extract_points(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """
        Extract curve points from segmentation mask.
        
        Args:
            mask: Binary segmentation mask
            
        Returns:
            List of (x, y) curve points
        """
        # Find skeleton of the mask for thin curve representation
        from cv2 import ximgproc
        try:
            skeleton = ximgproc.thinning(mask)
        except AttributeError:
            # Fallback if ximgproc not available
            skeleton = self._morphological_skeleton(mask)
        
        # Get non-zero points
        coords = np.column_stack(np.where(skeleton > 0))
        
        if len(coords) == 0:
            # Fallback to raw mask center line
            coords = np.column_stack(np.where(mask > 0))
            if len(coords) == 0:
                return []
        
        # Convert from (row, col) to (x, y)
        points = [(int(c[1]), int(c[0])) for c in coords]
        
        # Sort by x coordinate
        points.sort(key=lambda p: p[0])
        
        # Remove duplicate x values (keep median y)
        unique_points = {}
        for x, y in points:
            if x not in unique_points:
                unique_points[x] = []
            unique_points[x].append(y)
        
        result = []
        for x in sorted(unique_points.keys()):
            y = int(np.median(unique_points[x]))
            result.append((x, y))
        
        return result
    
    def _morphological_skeleton(self, mask: np.ndarray) -> np.ndarray:
        """
        Compute morphological skeleton as fallback.
        
        Args:
            mask: Binary mask
            
        Returns:
            Skeletonized mask
        """
        skeleton = np.zeros_like(mask)
        temp = mask.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        
        while True:
            eroded = cv2.erode(temp, kernel)
            opened = cv2.dilate(eroded, kernel)
            diff = cv2.subtract(temp, opened)
            skeleton = cv2.bitwise_or(skeleton, diff)
            temp = eroded.copy()
            
            if cv2.countNonZero(temp) == 0:
                break
        
        return skeleton
    
    def save_model(self, path: str) -> None:
        """
        Save the current model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model:
            self.model.save(path)
            logger.info(f"Model saved to {path}")


# Async wrapper for model warming
async def warm_model():
    """Pre-warm the U-Net model for faster first inference."""
    from config import settings
    
    if not TF_AVAILABLE:
        return
    
    segmenter = UNetSegmenter(model_path=settings.unet_model_path)
    segmenter.load_model()
    
    # Run a dummy inference to warm up
    dummy_input = np.zeros((256, 256), dtype=np.uint8)
    segmenter.segment(dummy_input)
    
    logger.info("U-Net model warmed up")
