"""
stats.py — Quantitative metrics for image differences.

Computes aggregate statistics from the detection results: SSIM score,
change percentage, pixel counts, region counts, and per-region sizes.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class DiffStatistics:
    """Aggregate statistics for a pair of compared images.

    Attributes
    ----------
    ssim_score : float
        Structural Similarity Index (``-1`` to ``1``; ``1`` = identical).
    changed_pixels : int
        Number of non-zero pixels in the binary difference mask.
    total_pixels : int
        Total number of pixels in the image.
    percentage_changed : float
        ``changed_pixels / total_pixels * 100``.
    num_regions : int
        Number of significant changed regions (contours).
    largest_region_area : int
        Area (in pixels) of the largest changed region, or ``0``.
    average_region_area : float
        Mean area across all regions, or ``0.0``.
    """

    ssim_score: float
    changed_pixels: int
    total_pixels: int
    percentage_changed: float
    num_regions: int
    largest_region_area: int
    average_region_area: float


def compute_statistics(
    ssim_score: float,
    thresh_mask: np.ndarray,
    contours: tuple,
) -> DiffStatistics:
    """Compute aggregate difference statistics.

    Parameters
    ----------
    ssim_score:
        The mean SSIM score returned by ``compute_ssim``.
    thresh_mask:
        Binary ``uint8`` mask (0 / 255) from the detection step.
    contours:
        Tuple of OpenCV contour arrays (only significant contours,
        as filtered by ``find_contours``).

    Returns
    -------
    DiffStatistics
        Frozen dataclass with all computed metrics.

    Raises
    ------
    ValueError
        If *thresh_mask* is ``None`` or empty.
    """
    if thresh_mask is None or thresh_mask.size == 0:
        raise ValueError("thresh_mask is empty or None.")

    total_pixels: int = int(thresh_mask.shape[0] * thresh_mask.shape[1])
    changed_pixels: int = int(cv2.countNonZero(thresh_mask))
    percentage_changed: float = (changed_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

    # Per-region areas.
    areas: list[int] = [int(cv2.contourArea(cnt)) for cnt in contours]

    num_regions: int = len(areas)
    largest_region_area: int = max(areas) if areas else 0
    average_region_area: float = float(np.mean(areas)) if areas else 0.0

    return DiffStatistics(
        ssim_score=round(ssim_score, 4),
        changed_pixels=changed_pixels,
        total_pixels=total_pixels,
        percentage_changed=round(percentage_changed, 2),
        num_regions=num_regions,
        largest_region_area=largest_region_area,
        average_region_area=round(average_region_area, 1),
    )
