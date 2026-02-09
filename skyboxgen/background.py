from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

from .projections import reproject_background


def destar_background(
    img: Image.Image,
    percentile: float = 99.7,
    blur_radius: float = 2.0,
    mask_expand: int = 1,
    max_mask: Optional[int] = None,
) -> Image.Image:
    gray = img.convert("L")
    blur = gray.filter(ImageFilter.GaussianBlur(blur_radius))
    gray_arr = np.asarray(gray, dtype=np.float32)
    blur_arr = np.asarray(blur, dtype=np.float32)
    high = np.clip(gray_arr - blur_arr, 0.0, 255.0)
    thresh = np.percentile(high, percentile)
    mask = high >= thresh
    if max_mask is not None:
        if mask.sum() > max_mask:
            return img
    mask_img = Image.fromarray((mask * 255).astype("uint8"), mode="L")
    if mask_expand > 0:
        size = mask_expand * 2 + 1
        mask_img = mask_img.filter(ImageFilter.MaxFilter(size=size))
    blur_color = img.filter(ImageFilter.GaussianBlur(blur_radius * 2))
    return Image.composite(blur_color, img, mask_img)


def preprocess_background(
    input_path: str,
    output_path: str,
    out_w: int,
    out_h: int,
    projection: str,
    fill: str,
    gain: float,
    destar: bool,
    destar_percentile: float,
    destar_blur: float,
    destar_expand: int,
    border_mask: bool,
    border_tolerance: float,
) -> None:
    # Convert a source all-sky background to equirectangular with optional cleanup.
    img = Image.open(input_path).convert("RGB")
    if destar:
        img = destar_background(img, destar_percentile, destar_blur, destar_expand)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = reproject_background(arr, out_w, out_h, projection, fill=fill)
    if border_mask:
        arr = _mask_border_color(arr, border_tolerance)
    if gain != 1.0:
        arr = np.clip(arr * gain, 0.0, 1.0)
    out = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")
    out.save(output_path)


def _mask_border_color(arr: np.ndarray, tolerance: float) -> np.ndarray:
    # Treat the dominant corner color as background and zero it out.
    h, w, _ = arr.shape
    corners = np.array(
        [
            arr[0, 0],
            arr[0, w - 1],
            arr[h - 1, 0],
            arr[h - 1, w - 1],
        ]
    )
    border_color = np.mean(corners, axis=0)
    diff = np.linalg.norm(arr - border_color, axis=2)
    mask = diff <= tolerance
    arr = arr.copy()
    arr[mask] = 0.0
    return arr
