"""
visualize.py — Visualization utilities for detected differences.

All functions return **RGB** ``uint8`` NumPy arrays suitable for
direct display with ``st.image()``.
"""

from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------


def create_heatmap(
    diff_map: np.ndarray,
    base_image: np.ndarray | None = None,
    *,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Generate a coloured heatmap from a difference map.

    Parameters
    ----------
    diff_map:
        Float64 per-pixel SSIM map (range ``[0, 1]``, where ``1`` =
        identical) **or** a ``uint8`` absolute-difference image.
    base_image:
        Optional BGR base image to blend the heatmap on top of.  If
        ``None``, the heatmap is returned on its own.
    alpha:
        Blend weight for the heatmap when *base_image* is provided.
    colormap:
        OpenCV colormap constant (default ``COLORMAP_JET``).

    Returns
    -------
    np.ndarray
        3-channel **RGB** ``uint8`` image.
    """
    if diff_map is None or diff_map.size == 0:
        raise ValueError("diff_map is empty or None.")

    # Normalise to uint8 if needed.
    if diff_map.dtype != np.uint8:
        # For SSIM maps: invert so larger = more change, then scale.
        if diff_map.max() <= 1.0:
            diff_uint8 = ((1.0 - diff_map) * 255).astype(np.uint8)
        else:
            diff_uint8 = cv2.normalize(
                diff_map, None, 0, 255, cv2.NORM_MINMAX,
            ).astype(np.uint8)
    else:
        diff_uint8 = diff_map

    heatmap_bgr = cv2.applyColorMap(diff_uint8, colormap)

    if base_image is not None:
        # Resize heatmap to match base if necessary.
        if heatmap_bgr.shape[:2] != base_image.shape[:2]:
            heatmap_bgr = cv2.resize(
                heatmap_bgr,
                (base_image.shape[1], base_image.shape[0]),
            )
        blended = cv2.addWeighted(base_image, 1 - alpha, heatmap_bgr, alpha, 0)
        return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Bounding boxes
# ---------------------------------------------------------------------------


def draw_bounding_boxes(
    image: np.ndarray,
    bounding_boxes: list[tuple[int, int, int, int]],
    *,
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw red rectangles around each changed region.

    Parameters
    ----------
    image:
        BGR ``uint8`` image to annotate.
    bounding_boxes:
        List of ``(x, y, w, h)`` tuples as returned by
        :func:`detection.find_contours`.
    color:
        BGR colour for the rectangles (default red).
    thickness:
        Line thickness in pixels.

    Returns
    -------
    np.ndarray
        Annotated **RGB** ``uint8`` image (copy of original).
    """
    if image is None or image.size == 0:
        raise ValueError("Image is empty or None.")

    annotated = image.copy()

    for x, y, w, h in bounding_boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Side-by-side comparison
# ---------------------------------------------------------------------------


def side_by_side(
    img1: np.ndarray,
    img2: np.ndarray,
    *,
    divider_width: int = 4,
    divider_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Create a horizontal side-by-side comparison of two images.

    Both images are resized to the same height (the larger of the two)
    before concatenation.

    Parameters
    ----------
    img1, img2:
        BGR or RGB ``uint8`` images.
    divider_width:
        Width (in pixels) of the vertical divider between the images.
    divider_color:
        BGR/RGB colour for the divider stripe.

    Returns
    -------
    np.ndarray
        Horizontally concatenated **RGB** ``uint8`` image.
    """
    if img1 is None or img1.size == 0:
        raise ValueError("First image is empty or None.")
    if img2 is None or img2.size == 0:
        raise ValueError("Second image is empty or None.")

    # Ensure both are 3-channel.
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Match heights.
    h1, h2 = img1.shape[0], img2.shape[0]
    target_h = max(h1, h2)

    if h1 != target_h:
        scale = target_h / h1
        img1 = cv2.resize(
            img1, (int(img1.shape[1] * scale), target_h),
        )
    if h2 != target_h:
        scale = target_h / h2
        img2 = cv2.resize(
            img2, (int(img2.shape[1] * scale), target_h),
        )

    # Build divider.
    divider = np.full(
        (target_h, divider_width, 3), divider_color, dtype=np.uint8,
    )

    combined = np.hstack([img1, divider, img2])

    # Return as RGB.
    return cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
