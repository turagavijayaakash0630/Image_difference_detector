"""
detection — Image difference detection algorithms.

Public API
----------
compute_ssim        Compare two grayscale images via SSIM.
compute_pixel_diff  Absolute pixel difference with adaptive threshold.
find_contours       Extract bounding boxes from a binary mask.
SSIMResult          Dataclass holding SSIM score, diff map, and mask.
ContourResult       Dataclass holding contours and bounding boxes.
"""

from detection.difference import (
    ContourResult,
    SSIMResult,
    compute_pixel_diff,
    compute_ssim,
    find_contours,
)

__all__: list[str] = [
    "compute_ssim",
    "compute_pixel_diff",
    "find_contours",
    "SSIMResult",
    "ContourResult",
]
