"""
image_utils.py
Utility functions for image loading, preprocessing, and conversion.
"""

import cv2
import numpy as np
from PIL import Image
import io


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert a PIL Image (RGB) to an OpenCV BGR numpy array."""
    rgb = np.array(pil_image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def cv2_to_pil(bgr_image: np.ndarray) -> Image.Image:
    """Convert an OpenCV BGR numpy array to a PIL Image (RGB)."""
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    """Convert raw image bytes to an OpenCV BGR array."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def resize_keep_aspect(image: np.ndarray, max_side: int = 640) -> np.ndarray:
    """Resize image so the longest side is at most max_side pixels."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return image
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def auto_orient(pil_image: Image.Image) -> Image.Image:
    """Auto-rotate image based on EXIF orientation tag."""
    try:
        from PIL import ImageOps
        return ImageOps.exif_transpose(pil_image)
    except Exception:
        return pil_image
