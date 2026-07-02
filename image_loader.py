"""
image_loader.py — Image loading and preprocessing utilities.

Provides functions to load uploaded images, validate formats, resize images
to matching dimensions, and perform colour-space conversions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

# Formats accepted by the application (checked against the MIME subtype).
_SUPPORTED_EXTENSIONS: set[str] = {"jpg", "jpeg", "png"}


def _validate_format(uploaded_file: UploadedFile) -> None:
    """Raise ``ValueError`` if the uploaded file is not a supported image.

    Parameters
    ----------
    uploaded_file:
        A Streamlit ``UploadedFile`` instance whose ``.type`` attribute
        contains the MIME type (e.g. ``"image/png"``).

    Raises
    ------
    ValueError
        If the file's MIME subtype is not in ``_SUPPORTED_EXTENSIONS``.
    """
    if uploaded_file.type is None:
        raise ValueError("Uploaded file has no MIME type. Please upload a JPG or PNG image.")

    # MIME type looks like "image/png", "image/jpeg", etc.
    mime_subtype = uploaded_file.type.split("/")[-1].lower()
    if mime_subtype not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format '{mime_subtype}'. "
            f"Accepted formats: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}."
        )


def load_image(uploaded_file: UploadedFile) -> np.ndarray:
    """Decode a Streamlit ``UploadedFile`` into a BGR NumPy array.

    The function validates the file format, reads the raw bytes, and uses
    OpenCV to decode the image.

    Parameters
    ----------
    uploaded_file:
        A file uploaded via ``st.file_uploader``.

    Returns
    -------
    np.ndarray
        The decoded image as a 3-channel BGR ``uint8`` array.

    Raises
    ------
    ValueError
        If the format is unsupported or decoding fails.
    """
    _validate_format(uploaded_file)

    # Read as raw bytes and convert to a NumPy byte array.
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(
            "Failed to decode the uploaded image. "
            "The file may be corrupted or empty."
        )

    return image


def resize_to_match(
    img_reference: np.ndarray,
    img_target: np.ndarray,
    *,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Resize ``img_target`` so its dimensions match ``img_reference``.

    The function uses ``cv2.resize`` with the specified interpolation method.
    ``cv2.INTER_AREA`` is the default because it produces the best results
    when *shrinking* images; for upscaling ``cv2.INTER_CUBIC`` is a good
    alternative that callers can pass explicitly.

    Parameters
    ----------
    img_reference:
        The reference image whose (height, width) determines the output size.
    img_target:
        The image to resize.
    interpolation:
        OpenCV interpolation flag (default ``cv2.INTER_AREA``).

    Returns
    -------
    np.ndarray
        A resized copy of ``img_target`` with the same (H, W) as
        ``img_reference``.

    Raises
    ------
    ValueError
        If either input is not a valid image array.
    """
    if img_reference is None or img_reference.size == 0:
        raise ValueError("Reference image is empty or None.")
    if img_target is None or img_target.size == 0:
        raise ValueError("Target image is empty or None.")

    ref_h, ref_w = img_reference.shape[:2]
    return cv2.resize(img_target, (ref_w, ref_h), interpolation=interpolation)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to single-channel grayscale.

    If the image is already single-channel it is returned unchanged.

    Parameters
    ----------
    image:
        A BGR or grayscale ``uint8`` NumPy array.

    Returns
    -------
    np.ndarray
        A 2-D (H × W) grayscale image.

    Raises
    ------
    ValueError
        If the image is ``None`` or empty.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot convert an empty or None image to grayscale.")

    # Already grayscale.
    if image.ndim == 2:
        return image

    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR image to RGB.

    Parameters
    ----------
    image:
        A 3-channel BGR ``uint8`` NumPy array.

    Returns
    -------
    np.ndarray
        The same image with channels reordered to RGB.

    Raises
    ------
    ValueError
        If the image is ``None``, empty, or not 3-channel.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot convert an empty or None image.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected a 3-channel image for BGR → RGB conversion, "
            f"got shape {image.shape}."
        )

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an RGB image to BGR.

    Parameters
    ----------
    image:
        A 3-channel RGB ``uint8`` NumPy array.

    Returns
    -------
    np.ndarray
        The same image with channels reordered to BGR.

    Raises
    ------
    ValueError
        If the image is ``None``, empty, or not 3-channel.
    """
    if image is None or image.size == 0:
        raise ValueError("Cannot convert an empty or None image.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Expected a 3-channel image for RGB → BGR conversion, "
            f"got shape {image.shape}."
        )

    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
