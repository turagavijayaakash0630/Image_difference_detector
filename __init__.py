"""
preprocessing — Image loading, validation, and transformation utilities.

Public API
----------
load_image        Load a Streamlit UploadedFile into a BGR NumPy array.
resize_to_match   Resize one image to match another's dimensions.
to_grayscale      Convert a BGR image to grayscale.
bgr_to_rgb        Convert BGR → RGB.
rgb_to_bgr        Convert RGB → BGR.
"""

from preprocessing.image_loader import (
    bgr_to_rgb,
    load_image,
    resize_to_match,
    rgb_to_bgr,
    to_grayscale,
)

__all__: list[str] = [
    "load_image",
    "resize_to_match",
    "to_grayscale",
    "bgr_to_rgb",
    "rgb_to_bgr",
]
