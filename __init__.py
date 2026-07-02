"""
visualization — Visual output utilities for detected differences.

Public API
----------
create_heatmap        Coloured heatmap (optionally blended onto base image).
draw_bounding_boxes   Red rectangles around changed regions.
side_by_side          Horizontal side-by-side comparison.
"""

from visualization.visualize import (
    create_heatmap,
    draw_bounding_boxes,
    side_by_side,
)

__all__: list[str] = [
    "create_heatmap",
    "draw_bounding_boxes",
    "side_by_side",
]
