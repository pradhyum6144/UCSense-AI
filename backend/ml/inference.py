"""
ML Inference Module

Provides model loading and inference utilities for the U-Net segmentation model.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


async def warm_model() -> None:
    """
    Pre-warm the ML model for faster first inference.
    
    This function is called during application startup to load
    the model into memory and run a warmup inference.
    """
    from config import settings
    
    try:
        # Check if TensorFlow is available
        import tensorflow as tf
        logger.info("TensorFlow available, warming U-Net model")
        
        from extraction.unet_segmenter import UNetSegmenter
        import numpy as np
        
        # Initialize segmenter
        segmenter = UNetSegmenter(model_path=settings.unet_model_path)
        
        # Try to load model
        if segmenter.load_model():
            # Run warmup inference with dummy data
            dummy_image = np.zeros((256, 256), dtype=np.uint8)
            _ = segmenter.segment(dummy_image)
            logger.info("U-Net model warmed successfully")
        else:
            logger.warning("Could not load U-Net model, using default architecture")
            
    except ImportError:
        logger.warning("TensorFlow not installed, U-Net segmentation unavailable")
    except Exception as e:
        logger.error(f"Model warming failed: {e}")


def load_model(model_path: str) -> Optional[object]:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model or None if loading failed
    """
    try:
        import tensorflow as tf
        
        if not Path(model_path).exists():
            logger.warning(f"Model file not found: {model_path}")
            return None
        
        model = tf.keras.models.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
        return model
        
    except ImportError:
        logger.error("TensorFlow not available")
        return None
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return None
