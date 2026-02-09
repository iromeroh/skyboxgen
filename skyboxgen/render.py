import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .background import destar_background
from .color import bv_to_rgb
from .coords import equatorial_to_galactic, galactic_to_equatorial
from .projections import reproject_background


def render_skybox(
    arrays: Dict[str, np.ndarray],
    system_xyz: Tuple[float, float, float],
    width: int,
    height: int,
    mag_limit: float,
    background_path: Optional[str],
    background_projection: str,
    background_frame: str,
    star_frame: str,
    background_fill: str,
    rotate_deg: Tuple[float, float, float],
    splat_sigma: float,
    splat_min_sigma: float,
    use_abs_mag: bool,
    background_destar: bool,
    destar_percentile: float,
    destar_blur: float,
    destar_expand: int,
    background_gain: float,
    star_gain: float = 0.2,
    exclude_nearby_pc: float = 0.05,
    min_distance_pc: float = 0.0,
    max_distance_pc: float = 0.0,
    label_entries: Optional[List[Tuple[str, Optional[int], Tuple[float, float, float]]]] = None,
    label_color: str = "#70ff70",
    label_font_size: int = 18,
    label_include_hip: bool = True,
    label_alpha: int = 200,
    guide_enable: bool = False,
    guide_color: str = "#70ff70",
    guide_alpha: int = 140,
    guide_meridian_step: float = 15.0,
    guide_label_font_size: int = 14,
    guide_label_lat: float = 60.0,
    constellation_segments: Optional[List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]] = None,
    constellation_labels: Optional[List[Tuple[str, Tuple[float, float, float]]]] = None,
    constellation_star_labels: Optional[List[Tuple[str, Tuple[float, float, float]]]] = None,
    constellation_color: str = "#70ff70",
    constellation_alpha: int = 160,
    constellation_line_width: int = 1,
    constellation_label_font_size: int = 14,
    constellation_star_font_size: int = 12,
    constellation_star_alpha: int = 160,
    constellation_frame: str = "equatorial",
) -> Image.Image:
    base = np.zeros((height, width, 3), dtype=np.float32)
    if background_path:
        bg = Image.open(background_path).convert("RGB")
        if background_destar:
            bg = destar_background(bg, destar_percentile, destar_blur, destar_expand)
        bg_arr = np.asarray(bg, dtype=np.float32) / 255.0
        bg_arr = reproject_background(
            bg_arr, width, height, background_projection, fill=background_fill
        )
        if background_gain != 1.0:
            bg_arr = np.clip(bg_arr * background_gain, 0.0, 1.0)
        base = np.clip(bg_arr, 0.0, 1.0).astype(np.float32)

    x = arrays["x"].astype(np.float64)
    y = arrays["y"].astype(np.float64)
    z = arrays["z"].astype(np.float64)
    absmag = arrays["absmag"].astype(np.float64)
    mag = arrays["mag"].astype(np.float64)
    ci = arrays["ci"].astype(np.float64)

    sx, sy, sz = system_xyz
    dx = x - sx
    dy = y - sy
    dz = z - sz
    dist = np.sqrt(dx * dx + dy * dy + dz * dz)
    dist[dist == 0] = np.nan
    dist_mask = np.ones_like(dist, dtype=bool)
    if exclude_nearby_pc > 0:
        dist_mask &= dist > exclude_nearby_pc
    if min_distance_pc > 0:
        dist_mask &= dist >= min_distance_pc
    if max_distance_pc > 0:
        dist_mask &= dist <= max_distance_pc

    if use_abs_mag:
        apparent = absmag + 5.0 * np.log10(dist) - 5.0
    else:
        apparent = mag
    visible = (apparent <= mag_limit) & dist_mask

    dx = dx[visible]
    dy = dy[visible]
    dz = dz[visible]
    apparent = apparent[visible]
    ci = ci[visible]
    dist = dist[visible]

    norm = np.sqrt(dx * dx + dy * dy + dz * dz)
    dx /= norm
    dy /= norm
    dz /= norm

    if star_frame != background_frame:
        vec = np.vstack([dx, dy, dz])
        if star_frame == "equatorial" and background_frame == "galactic":
            vec = equatorial_to_galactic(vec)
        elif star_frame == "galactic" and background_frame == "equatorial":
            vec = galactic_to_equatorial(vec)
        dx, dy, dz = vec[0], vec[1], vec[2]

    lon = np.arctan2(dy, dx)
    lat = np.arcsin(dz)
    lon, lat = _apply_rotation(lon, lat, rotate_deg)

    x_pix = (lon + math.pi) / (2 * math.pi) * width
    y_pix = (math.pi / 2 - lat) / math.pi * height

    intensity = np.power(10.0, -0.4 * (apparent - mag_limit)) * star_gain
    intensity = np.clip(intensity, 0.0, 1.0)

    for i in range(len(x_pix)):
        px = x_pix[i]
        py = y_pix[i]
        if not (0 <= px < width and 0 <= py < height):
            continue
        color = bv_to_rgb(ci[i])
        sigma = max(splat_min_sigma, splat_sigma * math.pow(10.0, -0.2 * apparent[i]))
        _add_gaussian(base, px, py, color, intensity[i], sigma)

    img = np.clip(base * 255.0, 0, 255).astype(np.uint8)
    out = Image.fromarray(img, mode="RGB")

    if label_entries or guide_enable or constellation_segments:
        out = out.convert("RGBA")
        draw = ImageDraw.Draw(out, "RGBA")
        if guide_enable:
            _draw_guides(
                draw,
                width,
                height,
                guide_color,
                guide_alpha,
                guide_meridian_step,
                guide_label_font_size,
                guide_label_lat,
            )
        if constellation_segments:
            _draw_constellations(
                draw,
                constellation_segments,
                constellation_labels or [],
                constellation_star_labels or [],
                system_xyz,
                width,
                height,
                star_frame,
                background_frame,
                constellation_color,
                constellation_alpha,
                constellation_line_width,
                constellation_label_font_size,
                constellation_star_font_size,
                constellation_star_alpha,
                constellation_frame,
            )
        if label_entries:
            _draw_labels(
                draw,
                label_entries,
                system_xyz,
                width,
                height,
                star_frame,
                background_frame,
                label_color,
                label_font_size,
                label_include_hip,
                label_alpha,
            )
        out = out.convert("RGB")
    return out


def _apply_rotation(lon: np.ndarray, lat: np.ndarray, rotate_deg: Tuple[float, float, float]):
    if rotate_deg == (0.0, 0.0, 0.0):
        return lon, lat
    roll, pitch, yaw = [math.radians(v) for v in rotate_deg]
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    x, y, z = _rotate_z(x, y, z, yaw)
    x, y, z = _rotate_y(x, y, z, pitch)
    x, y, z = _rotate_x(x, y, z, roll)

    lon = np.arctan2(y, x)
    lat = np.arcsin(z)
    return lon, lat


def _rotate_x(x: np.ndarray, y: np.ndarray, z: np.ndarray, angle: float):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    y2 = y * cos_a - z * sin_a
    z2 = y * sin_a + z * cos_a
    return x, y2, z2


def _rotate_y(x: np.ndarray, y: np.ndarray, z: np.ndarray, angle: float):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x2 = x * cos_a + z * sin_a
    z2 = -x * sin_a + z * cos_a
    return x2, y, z2


def _rotate_z(x: np.ndarray, y: np.ndarray, z: np.ndarray, angle: float):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x2 = x * cos_a - y * sin_a
    y2 = x * sin_a + y * cos_a
    return x2, y2, z


def _add_gaussian(
    img: np.ndarray,
    px: float,
    py: float,
    color: Tuple[float, float, float],
    intensity: float,
    sigma: float,
) -> None:
    if sigma <= 0.75:
        ix = int(px)
        iy = int(py)
        if 0 <= ix < img.shape[1] and 0 <= iy < img.shape[0]:
            img[iy, ix, :] = np.clip(img[iy, ix, :] + intensity * np.array(color), 0, 1)
        return
    radius = int(max(2, sigma * 3))
    x0 = int(max(0, px - radius))
    x1 = int(min(img.shape[1] - 1, px + radius))
    y0 = int(max(0, py - radius))
    y1 = int(min(img.shape[0] - 1, py + radius))
    if x1 <= x0 or y1 <= y0:
        return
    xs = np.arange(x0, x1 + 1)
    ys = np.arange(y0, y1 + 1)
    xv, yv = np.meshgrid(xs, ys)
    dx = xv - px
    dy = yv - py
    kernel = np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))
    for c in range(3):
        img[y0 : y1 + 1, x0 : x1 + 1, c] = np.clip(
            img[y0 : y1 + 1, x0 : x1 + 1, c] + kernel * intensity * color[c],
            0,
            1,
        )


def _draw_guides(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    color: str,
    alpha: int,
    meridian_step: float,
    label_font_size: int,
    label_lat: float,
) -> None:
    # Overlay galactic reference guides for orientation in the skybox.
    line_color = _parse_color(color, alpha)
    label_color = _parse_color(color, min(255, alpha + 40))
    font = _load_font(label_font_size)
    # Equator and tropics (galactic latitude).
    for lat, label in [(0.0, "Equator"), (23.5, "Tropic +23.5"), (-23.5, "Tropic -23.5")]:
        y = (math.pi / 2 - math.radians(lat)) / math.pi * height
        draw.line([(0, y), (width, y)], fill=line_color, width=1)
        draw.text((6, max(0, y - label_font_size - 2)), label, fill=label_color, font=font)
    # Meridian lines to the poles.
    step = max(1.0, meridian_step)
    lon = -180.0
    label_y = (math.pi / 2 - math.radians(label_lat)) / math.pi * height
    while lon < 180.0:
        x = (math.radians(lon) + math.pi) / (2 * math.pi) * width
        draw.line([(x, 0), (x, height)], fill=line_color, width=1)
        draw.text((x + 2, label_y), f"{int(lon)}°", fill=label_color, font=font)
        lon += step
    # Pole markers.
    _draw_pole_marker(draw, width / 2, 0, line_color)
    _draw_pole_marker(draw, width / 2, height, line_color)
    draw.text((width / 2 + 8, 6), "Galactic North", fill=label_color, font=font)
    draw.text((width / 2 + 8, height - label_font_size - 8), "Galactic South", fill=label_color, font=font)


def _draw_pole_marker(draw: ImageDraw.ImageDraw, x: float, y: float, color) -> None:
    r = 6
    draw.ellipse([(x - r, y - r), (x + r, y + r)], outline=color, width=1)


def _draw_labels(
    draw: ImageDraw.ImageDraw,
    label_entries: List[Tuple[str, Optional[int], Tuple[float, float, float]]],
    system_xyz: Tuple[float, float, float],
    width: int,
    height: int,
    star_frame: str,
    background_frame: str,
    color: str,
    font_size: int,
    include_hip: bool,
    alpha: int,
) -> None:
    # Label game stars as seen from the current system, for quick in-sky identification.
    font = _load_font(font_size)
    text_color = _parse_color(color, alpha)
    dot_color = _parse_color(color, min(255, alpha + 40))
    for name, hip, xyz in label_entries:
        dx = xyz[0] - system_xyz[0]
        dy = xyz[1] - system_xyz[1]
        dz = xyz[2] - system_xyz[2]
        norm = math.sqrt(dx * dx + dy * dy + dz * dz)
        if norm == 0:
            continue
        dx /= norm
        dy /= norm
        dz /= norm
        vec = np.array([dx, dy, dz], dtype=np.float64)
        if star_frame != background_frame:
            if star_frame == "equatorial" and background_frame == "galactic":
                vec = equatorial_to_galactic(vec)
            elif star_frame == "galactic" and background_frame == "equatorial":
                vec = galactic_to_equatorial(vec)
        lon = math.atan2(vec[1], vec[0])
        lat = math.asin(vec[2])
        x = (lon + math.pi) / (2 * math.pi) * width
        y = (math.pi / 2 - lat) / math.pi * height
        label = name
        if include_hip and hip:
            label = f"{label} (HIP {hip})"
        draw.ellipse([(x - 2, y - 2), (x + 2, y + 2)], fill=dot_color)
        draw.text((x + 4, y + 4), label, fill=text_color, font=font)


def _parse_color(color: str, alpha: int) -> Tuple[int, int, int, int]:
    color = color.lstrip("#")
    if len(color) == 6:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
    else:
        r, g, b = 0, 255, 0
    return (r, g, b, max(0, min(255, alpha)))


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    # Use a known sans font when available; fall back to PIL's default bitmap font.
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_constellations(
    draw: ImageDraw.ImageDraw,
    segments: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]],
    const_labels: List[Tuple[str, Tuple[float, float, float]]],
    star_labels: List[Tuple[str, Tuple[float, float, float]]],
    system_xyz: Tuple[float, float, float],
    width: int,
    height: int,
    star_frame: str,
    background_frame: str,
    color: str,
    alpha: int,
    line_width: int,
    label_font_size: int,
    star_font_size: int,
    star_alpha: int,
    constellation_frame: str,
) -> None:
    # Draw constellation lines in the current system's sky with readable labels.
    line_color = _parse_color(color, alpha)
    label_color = _parse_color(color, min(255, alpha + 40))
    star_color = _parse_color(color, star_alpha)
    label_font = _load_font(label_font_size)
    star_font = _load_font(star_font_size)
    for a, b in segments:
        _draw_segment_samples(
            draw,
            a,
            b,
            system_xyz,
            width,
            height,
            star_frame,
            constellation_frame,
            line_color,
            line_width,
        )
    for text, xyz in const_labels:
        _, _, x, y = _project_xyz(xyz, system_xyz, width, height, star_frame, constellation_frame)
        draw.text((x + 4, y + 4), text, fill=label_color, font=label_font)
    for text, xyz in star_labels:
        _, _, x, y = _project_xyz(xyz, system_xyz, width, height, star_frame, constellation_frame)
        draw.text((x + 3, y + 3), text, fill=star_color, font=star_font)


def _project_xyz(
    xyz: Tuple[float, float, float],
    system_xyz: Tuple[float, float, float],
    width: int,
    height: int,
    star_frame: str,
    background_frame: str,
) -> Tuple[float, float, float, float]:
    dx = xyz[0] - system_xyz[0]
    dy = xyz[1] - system_xyz[1]
    dz = xyz[2] - system_xyz[2]
    norm = math.sqrt(dx * dx + dy * dy + dz * dz)
    if norm == 0:
        return (0.0, 0.0, 0.0, 0.0)
    dx /= norm
    dy /= norm
    dz /= norm
    vec = np.array([dx, dy, dz], dtype=np.float64)
    if star_frame != background_frame:
        if star_frame == "equatorial" and background_frame == "galactic":
            vec = equatorial_to_galactic(vec)
        elif star_frame == "galactic" and background_frame == "equatorial":
            vec = galactic_to_equatorial(vec)
    lon = math.atan2(vec[1], vec[0])
    lat = math.asin(vec[2])
    x = (lon + math.pi) / (2 * math.pi) * width
    y = (math.pi / 2 - lat) / math.pi * height
    return (lon, lat, x, y)


def _draw_wrapped_line(
    draw: ImageDraw.ImageDraw,
    width: int,
    height: int,
    color: Tuple[int, int, int, int],
    line_width: int,
    lon1: float,
    lat1: float,
    x1: float,
    y1: float,
    lon2: float,
    lat2: float,
    x2: float,
    y2: float,
) -> None:
    # Avoid seam-crossing lines by splitting at the wrap boundary.
    if abs(x1 - x2) <= width / 2:
        draw.line([(x1, y1), (x2, y2)], fill=color, width=line_width)
        return
    lon1w = lon1 if lon1 >= 0 else lon1 + 2 * math.pi
    lon2w = lon2 if lon2 >= 0 else lon2 + 2 * math.pi
    if lon2w < lon1w:
        lon2w += 2 * math.pi
    t = (2 * math.pi - lon1w) / (lon2w - lon1w)
    lat_seam = lat1 + (lat2 - lat1) * t
    y_seam = (math.pi / 2 - lat_seam) / math.pi * height
    x_seam = width
    draw.line([(x1, y1), (x_seam, y_seam)], fill=color, width=line_width)
    draw.line([(0, y_seam), (x2, y2)], fill=color, width=line_width)


def _draw_segment_samples(
    draw: ImageDraw.ImageDraw,
    a: Tuple[float, float, float],
    b: Tuple[float, float, float],
    system_xyz: Tuple[float, float, float],
    width: int,
    height: int,
    star_frame: str,
    constellation_frame: str,
    color: Tuple[int, int, int, int],
    line_width: int,
) -> None:
    # Subdivide lines so seam crossings and pole regions render without wrap artifacts.
    steps = 24
    prev = None
    for i in range(steps + 1):
        t = i / steps
        xyz = (
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        )
        lon, lat, x, y = _project_xyz(
            xyz, system_xyz, width, height, star_frame, constellation_frame
        )
        if prev is not None:
            lon2, lat2, x2, y2 = prev
            _draw_wrapped_line(
                draw, width, height, color, line_width, lon2, lat2, x2, y2, lon, lat, x, y
            )
        prev = (lon, lat, x, y)
