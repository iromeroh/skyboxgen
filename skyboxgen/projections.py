import math
from typing import Literal

import numpy as np


def reproject_background(
    src: np.ndarray,
    out_w: int,
    out_h: int,
    projection: Literal["equirectangular", "mollweide"],
    fill: Literal["wrap", "black"] = "wrap",
) -> np.ndarray:
    if projection == "equirectangular":
        return _resize_nearest(src, out_w, out_h)
    if projection == "mollweide":
        return _mollweide_to_equirectangular(src, out_w, out_h, fill)
    raise ValueError(f"Unsupported projection: {projection}")


def _resize_nearest(src: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    src_h, src_w, _ = src.shape
    xs = (np.linspace(0, src_w - 1, out_w)).astype(np.int32)
    ys = (np.linspace(0, src_h - 1, out_h)).astype(np.int32)
    return src[ys[:, None], xs[None, :], :]


def _mollweide_to_equirectangular(
    src: np.ndarray, out_w: int, out_h: int, fill: Literal["wrap", "black"]
) -> np.ndarray:
    src_h, src_w, _ = src.shape
    lon = np.linspace(-math.pi, math.pi, out_w, endpoint=False)
    lat = np.linspace(math.pi / 2, -math.pi / 2, out_h)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    theta = _solve_mollweide_theta(lat_grid)
    x_m = (2 * math.sqrt(2) / math.pi) * lon_grid * np.cos(theta)
    y_m = math.sqrt(2) * np.sin(theta)

    x_norm = x_m / (4 * math.sqrt(2)) + 0.5
    y_norm = 0.5 - (y_m / (2 * math.sqrt(2)))
    u = x_norm * (src_w - 1)
    v = y_norm * (src_h - 1)

    ellipse = (x_m / (2 * math.sqrt(2))) ** 2 + (y_m / math.sqrt(2)) ** 2 <= 1.0
    if fill == "wrap":
        u_safe = np.clip(u, 0, src_w - 1)
        v_safe = np.clip(v, 0, src_h - 1)
        out = _bilinear_sample(src, u_safe.reshape(-1), v_safe.reshape(-1)).reshape(
            out_h, out_w, 3
        )
    else:
        out = np.zeros((out_h, out_w, 3), dtype=src.dtype)
        u_safe = np.clip(u[ellipse], 0, src_w - 1)
        v_safe = np.clip(v[ellipse], 0, src_h - 1)
        out[ellipse] = _bilinear_sample(src, u_safe, v_safe)
    return out


def _solve_mollweide_theta(lat: np.ndarray) -> np.ndarray:
    theta = lat.copy()
    for _ in range(10):
        delta = (2 * theta + np.sin(2 * theta) - math.pi * np.sin(lat)) / (
            2 + 2 * np.cos(2 * theta)
        )
        theta -= delta
    return np.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)


def _bilinear_sample(src: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    x0 = np.floor(u).astype(np.int32)
    y0 = np.floor(v).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, src.shape[1] - 1)
    y1 = np.clip(y0 + 1, 0, src.shape[0] - 1)
    u_ratio = u - x0
    v_ratio = v - y0
    top = (1 - u_ratio)[:, None] * src[y0, x0] + u_ratio[:, None] * src[y0, x1]
    bottom = (1 - u_ratio)[:, None] * src[y1, x0] + u_ratio[:, None] * src[y1, x1]
    return (1 - v_ratio)[:, None] * top + v_ratio[:, None] * bottom
