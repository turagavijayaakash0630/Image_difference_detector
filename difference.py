"""
difference.py — Core difference-detection algorithms.

Provides functions to compare two grayscale images using SSIM, compute
pixel-level difference masks, clean noise with morphological operations,
and extract bounding boxes around changed regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SSIMResult:
    """Container returned by :func:`compute_ssim`.

    Attributes
    ----------
    score : float
        The mean SSIM score in the range ``[−1, 1]``.  A value of ``1``
        means the images are identical.
    diff_map : np.ndarray
        Per-pixel SSIM map (float64, same shape as the input images).
        Lower values indicate more change.
    thresh_mask : np.ndarray
        Binary ``uint8`` mask (0 / 255) obtained by thresholding the
        inverted diff map.
    """

    score: float
    diff_map: np.ndarray
    thresh_mask: np.ndarray


@dataclass(frozen=True, slots=True)
class ContourResult:
    """Container returned by :func:`find_contours`.

    Attributes
    ----------
    contours : tuple
        Raw OpenCV contour arrays.
    bounding_boxes : list[tuple[int, int, int, int]]
        ``(x, y, w, h)`` rectangles for every significant contour.
    """

    contours: tuple
    bounding_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SSIM comparison
# ---------------------------------------------------------------------------


def compute_ssim(
    gray1: np.ndarray,
    gray2: np.ndarray,
    *,
    sensitivity: int = 50,
) -> SSIMResult:
    """Compare two grayscale images using Structural Similarity (SSIM).

    Parameters
    ----------
    gray1, gray2:
        Single-channel ``uint8`` images of identical shape.
    sensitivity:
        Integer ``0–100``.  Higher values make the detector more
        sensitive (i.e. the binary threshold is *lower*, so fainter
        differences are kept).  ``50`` corresponds to the classic
        Otsu auto-threshold.

    Returns
    -------
    SSIMResult
        Named container with ``score``, ``diff_map``, and ``thresh_mask``.

    Raises
    ------
    ValueError
        If images have different shapes or are not single-channel.
    """
    _validate_pair(gray1, gray2)

    score, diff_map = ssim(gray1, gray2, full=True)

    # diff_map is float64 in [0, 1] where 1 = identical.
    # Invert so that *larger* values indicate *more* change, then scale
    # to uint8 for thresholding.
    diff_uint8 = ((1.0 - diff_map) * 255).astype(np.uint8)

    thresh_mask = _apply_threshold(diff_uint8, sensitivity)
    thresh_mask = _morphological_clean(thresh_mask)

    return SSIMResult(score=float(score), diff_map=diff_map, thresh_mask=thresh_mask)


# ---------------------------------------------------------------------------
# Pixel-level absolute difference
# ---------------------------------------------------------------------------


def compute_pixel_diff(
    gray1: np.ndarray,
    gray2: np.ndarray,
    *,
    sensitivity: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the absolute pixel difference and a thresholded binary mask.

    Parameters
    ----------
    gray1, gray2:
        Single-channel ``uint8`` images of identical shape.
    sensitivity:
        ``0–100``.  Controls the binary threshold applied to the
        absolute difference image.

    Returns
    -------
    diff : np.ndarray
        Absolute difference image (``uint8``).
    mask : np.ndarray
        Binary ``uint8`` mask (0 / 255) after thresholding and
        morphological cleanup.

    Raises
    ------
    ValueError
        If images have different shapes or are not single-channel.
    """
    _validate_pair(gray1, gray2)

    diff = cv2.absdiff(gray1, gray2)
    mask = _apply_threshold(diff, sensitivity)
    mask = _morphological_clean(mask)

    return diff, mask


# ---------------------------------------------------------------------------
# Contour / bounding-box extraction
# ---------------------------------------------------------------------------


def find_contours(
    mask: np.ndarray,
    *,
    min_area: int = 100,
) -> ContourResult:
    """Find contours in a binary mask and return bounding boxes.

    Parameters
    ----------
    mask:
        Binary ``uint8`` image (0 / 255).
    min_area:
        Minimum contour area in pixels.  Contours smaller than this
        are discarded as noise.

    Returns
    -------
    ContourResult
        Contains raw contour arrays and their bounding boxes.

    Raises
    ------
    ValueError
        If *mask* is ``None`` or empty.
    """
    if mask is None or mask.size == 0:
        raise ValueError("Mask image is empty or None.")

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    bounding_boxes: list[tuple[int, int, int, int]] = [
        cv2.boundingRect(cnt)
        for cnt in contours
        if cv2.contourArea(cnt) >= min_area
    ]

    # Filter contours list to match (keep only significant ones).
    significant = tuple(
        cnt for cnt in contours if cv2.contourArea(cnt) >= min_area
    )

    return ContourResult(contours=significant, bounding_boxes=bounding_boxes)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_pair(img1: np.ndarray, img2: np.ndarray) -> None:
    """Raise ``ValueError`` if images aren't valid single-channel pairs."""
    if img1 is None or img1.size == 0:
        raise ValueError("First image is empty or None.")
    if img2 is None or img2.size == 0:
        raise ValueError("Second image is empty or None.")
    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError(
            "Both images must be single-channel (grayscale).  "
            f"Got ndim={img1.ndim} and ndim={img2.ndim}."
        )
    if img1.shape != img2.shape:
        raise ValueError(
            f"Image shapes must match.  Got {img1.shape} vs {img2.shape}."
        )


def _apply_threshold(diff_uint8: np.ndarray, sensitivity: int) -> np.ndarray:
    """Return a binary mask by thresholding *diff_uint8*.

    When ``sensitivity == 50`` Otsu's method is used.  Otherwise the
    threshold is linearly interpolated:  ``0`` → threshold 255
    (nothing detected), ``100`` → threshold 0 (everything detected).
    """
    sensitivity = max(0, min(100, sensitivity))

    if sensitivity == 50:
        _, mask = cv2.threshold(
            diff_uint8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
    else:
        # Map sensitivity 0→255, 100→0  (higher sensitivity = lower thresh).
        thresh_value = int(255 * (1 - sensitivity / 100))
        _, mask = cv2.threshold(diff_uint8, thresh_value, 255, cv2.THRESH_BINARY)

    return mask


def _morphological_clean(
    mask: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    """Remove noise from a binary mask using morphological opening + closing.

    Opening removes small white specks; closing fills small black holes.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size),
    )
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned
