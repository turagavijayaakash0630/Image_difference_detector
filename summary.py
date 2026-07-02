"""
summary.py — Text summarization of detected differences.

This module generates a human-readable summary describing the
differences detected between two images.
"""

from __future__ import annotations

from typing import List, Tuple


def get_region_locations(
    bounding_boxes: List[Tuple[int, int, int, int]],
    img_shape: Tuple[int, int],
) -> List[str]:
    """
    Determine approximate locations of changed regions.

    Parameters
    ----------
    bounding_boxes : list
        List of (x, y, w, h)

    img_shape : tuple
        (height, width)

    Returns
    -------
    list[str]
    """

    height, width = img_shape[:2]
    locations = []

    for x, y, w, h in bounding_boxes:
        cx = x + w / 2
        cy = y + h / 2

        horizontal = (
            "left"
            if cx < width / 3
            else "right"
            if cx > 2 * width / 3
            else "center"
        )

        vertical = (
            "top"
            if cy < height / 3
            else "bottom"
            if cy > 2 * height / 3
            else "center"
        )

        if horizontal == "center" and vertical == "center":
            locations.append("center")
        else:
            locations.append(f"{vertical}-{horizontal}")

    return list(set(locations))


def generate_summary(
    ssim_score: float,
    num_regions: int,
    changed_percentage: float,
    largest_region: float,
    bounding_boxes: List[Tuple[int, int, int, int]],
    img_shape: Tuple[int, int],
) -> str:
    """
    Generate a human-readable summary of detected differences.
    """

    if num_regions == 0 or changed_percentage < 0.1:
        return (
            f"The two images are nearly identical. "
            f"Structural Similarity Index (SSIM) is {ssim_score:.3f}. "
            "No significant visual differences were detected."
        )

    locations = get_region_locations(bounding_boxes, img_shape)

    location_text = ", ".join(locations)

    severity = (
        "minor"
        if changed_percentage < 5
        else "moderate"
        if changed_percentage < 15
        else "major"
    )

    summary = (
        f"The comparison detected {num_regions} changed region"
        f"{'' if num_regions == 1 else 's'}. "
        f"The overall SSIM score is {ssim_score:.3f}, "
        f"indicating {severity} visual differences. "
        f"Approximately {changed_percentage:.2f}% of the image has changed. "
        f"The largest detected region covers {largest_region:.0f} pixels. "
        f"Most changes are located around: {location_text}."
    )

    return summary