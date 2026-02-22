import json
import math
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

from skyboxgen.catalog import load_hyg_catalog, search_stars
from skyboxgen.constellation_lines import build_constellation_overlays
from skyboxgen.game_stars import load_game_stars, resolve_game_star, GameStar
from skyboxgen.render import render_skybox

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "out")
#OUT_DIR = "out\\"
OUT_DDS_DIR = os.path.join(ROOT, "out_dds")
#OUT_DDS_DIR = "out_dds\\"

_CATALOG_LOCK = threading.Lock()
_CATALOG = None
_GAME_STARS_LOCK = threading.Lock()
_GAME_STARS = None


def _load_catalog(path: str):
    global _CATALOG
    with _CATALOG_LOCK:
        if _CATALOG is None or _CATALOG.get("path") != path:
            _CATALOG = {"path": path, "catalog": load_hyg_catalog(path)}
        return _CATALOG["catalog"]


def _load_catalog_fresh(path: str):
    return load_hyg_catalog(path)


def _load_game_stars(path: str = "data/game_stars.csv") -> List[GameStar]:
    global _GAME_STARS
    with _GAME_STARS_LOCK:
        if _GAME_STARS is None or _GAME_STARS.get("path") != path:
            try:
                _GAME_STARS = {"path": path, "stars": load_game_stars(path)}
            except FileNotFoundError:
                _GAME_STARS = {"path": path, "stars": []}
        return _GAME_STARS["stars"]


def _read_json(request: BaseHTTPRequestHandler) -> Dict[str, Any]:
    length = int(request.headers.get("Content-Length", "0"))
    data = request.rfile.read(length) if length else b"{}"
    return json.loads(data.decode("utf-8"))


def _write_json(request: BaseHTTPRequestHandler, status: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    request.send_response(status)
    request.send_header("Content-Type", "application/json")
    request.send_header("Content-Length", str(len(body)))
    request.end_headers()
    request.wfile.write(body)


def _safe_path(base: str, rel: str) -> Optional[str]:
    target = os.path.abspath(os.path.join(base, rel))
    if not target.startswith(base):
        return None
    return target


def _get_calibration(params: Dict[str, Any]) -> Dict[str, Any]:
    calib = params.get("calibration") or {}
    return {
        "name": calib.get("name", "Alpha Centauri"),
        "hip": calib.get("hip"),
        "target": calib.get("target", {"x": -0.4952, "y": -0.4141, "z": -1.1566}),
        "scale": float(calib.get("scale", 1.0)),
        "rotation": calib.get("rotation", {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}),
    }


def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    rx = math.radians(roll)
    ry = math.radians(pitch)
    rz = math.radians(yaw)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def _calibration_transform(catalog, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, float]:
    calib = _get_calibration(params)
    target = calib["target"]
    scale = float(calib["scale"])
    rot = _rotation_matrix(
        calib["rotation"]["roll"],
        calib["rotation"]["pitch"],
        calib["rotation"]["yaw"],
    )
    ref_star = None
    if calib["hip"]:
        ref_star = catalog.by_hip.get(int(calib["hip"]))
    if not ref_star:
        matches = search_stars(catalog, calib["name"], limit=1)
        ref_star = matches[0] if matches else None
    if not ref_star:
        ref_star = catalog.stars[0]
    ref = np.array([ref_star.x, ref_star.y, ref_star.z], dtype=np.float64)
    t = np.array([target["x"], target["y"], target["z"]], dtype=np.float64) - scale * rot.dot(ref)
    return rot, t, scale


def _compute_calibration_scale(catalog, calib: Dict[str, Any]) -> float:
    """Compute scale factor from calibration reference star.

    Returns the ratio of target distance to HYG distance from Sol.
    Falls back to calib["scale"] or 1.0 if computation fails.
    """
    target = calib["target"]

    # Find reference star
    ref_star = None
    if calib.get("hip"):
        ref_star = catalog.by_hip.get(int(calib["hip"]))
    if not ref_star:
        # Try multiple name variants for Alpha Centauri
        name = calib.get("name", "")
        for search_name in [name, "Rigil Kentaurus", "Alpha Centauri"]:
            if search_name:
                matches = search_stars(catalog, search_name, limit=1)
                if matches:
                    ref_star = matches[0]
                    break

    if ref_star:
        ref_hyg = np.array([ref_star.x, ref_star.y, ref_star.z], dtype=np.float64)
        target_vec = np.array([target["x"], target["y"], target["z"]], dtype=np.float64)
        ref_dist = np.linalg.norm(ref_hyg)
        target_dist = np.linalg.norm(target_vec)
        if ref_dist > 1e-10 and target_dist > 1e-10:
            return target_dist / ref_dist

    # Fallback to configured scale or 1.0
    scale = float(calib.get("scale", 1.0))
    return scale if scale > 1e-10 else 1.0


def _game_to_hyg(catalog, params: Dict[str, Any], xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Convert game coordinates to HYG parsec coordinates.

    Sol is at origin (0,0,0) in both systems.
    """
    calib = _get_calibration(params)
    rot = _rotation_matrix(
        calib["rotation"]["roll"],
        calib["rotation"]["pitch"],
        calib["rotation"]["yaw"],
    )
    scale = _compute_calibration_scale(catalog, calib)

    # Inverse transform: hyg = inv(rot) * game / scale
    vec = np.array(xyz, dtype=np.float64)
    inv = np.linalg.inv(rot)
    hyg = inv.dot(vec) / scale
    return float(hyg[0]), float(hyg[1]), float(hyg[2])


def _hyg_to_game(catalog, params: Dict[str, Any], xyz: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Convert HYG parsec coordinates to calibrated game coordinates.

    Sol is kept at origin (0,0,0). The reference star is used to compute
    the scale factor so that distances from Sol match game coordinates.
    """
    calib = _get_calibration(params)
    rot = _rotation_matrix(
        calib["rotation"]["roll"],
        calib["rotation"]["pitch"],
        calib["rotation"]["yaw"],
    )
    scale = _compute_calibration_scale(catalog, calib)

    # Transform: game = scale * rot * hyg (no translation, Sol stays at origin)
    vec = np.array(xyz, dtype=np.float64)
    game = scale * rot.dot(vec)
    return float(game[0]), float(game[1]), float(game[2])


def _eq_to_gal(vec: np.ndarray) -> np.ndarray:
    mat = np.array(
        [
            [-0.0548755604, -0.8734370902, -0.4838350155],
            [0.4941094279, -0.4448296300, 0.7469822445],
            [-0.8676661490, -0.1980763734, 0.4559837762],
        ],
        dtype=np.float64,
    )
    return mat.dot(vec)


def _resolve_system_xyz(catalog, params: Dict[str, Any]) -> Tuple[str, Tuple[float, float, float]]:
    system = params.get("system") or {}
    name = system.get("name")
    hip = system.get("hip")
    xyz = system.get("xyz")
    if xyz:
        hyg = _game_to_hyg(catalog, params, (xyz["x"], xyz["y"], xyz["z"]))
        return name or "CustomSystem", hyg
    if hip:
        star = catalog.by_hip.get(int(hip))
        if star:
            return star.proper or star.bf or star.gl or f"HIP{star.hip}", (star.x, star.y, star.z)
    if name:
        matches = search_stars(catalog, name, limit=1)
        if matches:
            star = matches[0]
            return star.proper or star.bf or star.gl or f"HIP{star.hip}", (star.x, star.y, star.z)
    return "Sol", (0.0, 0.0, 0.0)


def _load_label_entries(catalog, params: Dict[str, Any], system_name: str):
    label_path = params.get("label_game_stars")
    if not label_path:
        return None
    entries = []
    for entry in load_game_stars(label_path):
        try:
            name, xyz, star = resolve_game_star(catalog, entry)
            if name.lower() == system_name.lower():
                continue
            hip = star.hip if star else None
            if entry.fictional and entry.x is not None:
                xyz = _game_to_hyg(catalog, params, (entry.x, entry.y, entry.z))
            entries.append((name, hip, xyz))
        except ValueError:
            continue
    return entries


def _constellation_overlays(catalog, params: Dict[str, Any]):
    if not params.get("overlay_constellations"):
        return (None, None, None)
    line_path = params.get("constellation_lines", "data/ConstellationLines.csv")
    segments, const_labels, star_labels = build_constellation_overlays(
        catalog,
        line_path,
        params.get("constellation_star_mag_limit", 2.5),
    )
    if not params.get("constellation_star_labels"):
        star_labels = []
    return segments, const_labels, star_labels


def _build_equator_labels(params: Dict[str, Any]) -> Optional[List[Tuple[str, Tuple[float, float, float]]]]:
    """Build equator label entries from request parameters."""
    label_text = params.get("equator_labels")
    if label_text:
        return [(label_text, (0.0, 0.0, 0.0))]
    return None


class SkyboxHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/viewer.html":
            self._serve_file(os.path.join(ROOT, "viewer.html"))
            return
        if parsed.path.startswith("/out/"):
            rel_path = unquote(parsed.path.replace("/out/", ""))
            target = _safe_path(OUT_DIR, rel_path)
            if target:
                self._serve_file(target)
            else:
                self.send_error(404)
            return
        if parsed.path.startswith("/out_dds/"):
            rel_path = unquote(parsed.path.replace("/out_dds/", ""))
            target = _safe_path(OUT_DDS_DIR, rel_path)
            if target:
                self._serve_file(target)
            else:
                self.send_error(404)
            return
        if parsed.path.startswith("/api/status"):
            _write_json(self, 200, {"ok": True})
            return
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/search":
            payload = _read_json(self)
            query = (payload.get("query") or "").strip()
            limit = int(payload.get("limit") or 10)
            include_game_stars = bool(payload.get("include_game_stars", False))
            catalog = _load_catalog(payload.get("catalog", "hygdata_v42.csv"))

            items = []
            seen_hips = set()

            # Search game stars first if requested
            if include_game_stars:
                game_stars = _load_game_stars()
                query_lower = query.lower()
                for gs in game_stars:
                    if query_lower in gs.game_name.lower():
                        # Try to resolve the game star to HYG
                        try:
                            name, xyz, hyg_star = resolve_game_star(catalog, gs)
                            if hyg_star:
                                hyg_name = hyg_star.proper or hyg_star.bf or hyg_star.gl or f"HIP{hyg_star.hip}"
                                # Calculate calibrated game coordinates
                                calibrated = None
                                if payload.get("calibration"):
                                    gx, gy, gz = _hyg_to_game(catalog, payload, (hyg_star.x, hyg_star.y, hyg_star.z))
                                    calibrated = {"x": gx, "y": gy, "z": gz}
                                item = {
                                    "name": hyg_name,
                                    "game_name": gs.game_name,
                                    "hip": hyg_star.hip,
                                    "x": hyg_star.x,
                                    "y": hyg_star.y,
                                    "z": hyg_star.z,
                                    "mag": hyg_star.mag,
                                    "spect": hyg_star.spect,
                                    "is_game_star": True,
                                    "calibrated_xyz": calibrated,
                                }
                                if hyg_star.hip not in seen_hips:
                                    items.append(item)
                                    seen_hips.add(hyg_star.hip)
                            elif gs.fictional and gs.x is not None:
                                # Fictional star with game coordinates
                                item = {
                                    "name": gs.game_name,
                                    "game_name": gs.game_name,
                                    "hip": None,
                                    "x": gs.x,
                                    "y": gs.y,
                                    "z": gs.z,
                                    "mag": None,
                                    "spect": None,
                                    "is_game_star": True,
                                    "fictional": True,
                                    "calibrated_xyz": {"x": gs.x, "y": gs.y, "z": gs.z} if gs.x is not None else None,
                                }
                                items.append(item)
                        except ValueError:
                            # Game star couldn't be resolved, skip
                            pass
                        if len(items) >= limit:
                            break

            # Search HYG catalog
            if len(items) < limit:
                matches = search_stars(catalog, query, limit=limit - len(items))
                for star in matches:
                    if star.hip in seen_hips:
                        continue
                    calibrated = None
                    if payload.get("calibration"):
                        gx, gy, gz = _hyg_to_game(catalog, payload, (star.x, star.y, star.z))
                        calibrated = {"x": gx, "y": gy, "z": gz}
                    items.append(
                        {
                            "name": star.proper or star.bf or star.gl or f"HIP{star.hip}",
                            "hip": star.hip,
                            "x": star.x,
                            "y": star.y,
                            "z": star.z,
                            "mag": star.mag,
                            "spect": star.spect,
                            "is_game_star": False,
                            "calibrated_xyz": calibrated,
                        }
                    )
                    if star.hip:
                        seen_hips.add(star.hip)

            _write_json(self, 200, {"results": items})
            return
        if parsed.path == "/api/calibrate_coords":
            payload = _read_json(self)
            catalog = _load_catalog(payload.get("catalog", "hygdata_v42.csv"))
            hyg_xyz = payload.get("hyg_xyz")
            if hyg_xyz:
                gx, gy, gz = _hyg_to_game(catalog, payload, (hyg_xyz["x"], hyg_xyz["y"], hyg_xyz["z"]))
                _write_json(self, 200, {"calibrated_xyz": {"x": gx, "y": gy, "z": gz}})
            else:
                _write_json(self, 400, {"error": "hyg_xyz required"})
            return
        if parsed.path == "/api/pick":
            payload = _read_json(self)
            catalog = _load_catalog(payload.get("catalog", "hygdata_v42.csv"))
            system_name, system_xyz = _resolve_system_xyz(catalog, payload)
            lon_deg = float(payload.get("lon_deg"))
            lat_deg = float(payload.get("lat_deg"))
            lon = math.radians(lon_deg)
            lat = math.radians(lat_deg)
            target = np.array([math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)])
            mag_limit = float(payload.get("mag_limit", 7.5))
            min_dist = float(payload.get("min_distance_ly", 0.0)) / 3.26156
            max_dist = float(payload.get("max_distance_ly", 0.0)) / 3.26156
            dx = catalog.arrays["x"] - system_xyz[0]
            dy = catalog.arrays["y"] - system_xyz[1]
            dz = catalog.arrays["z"] - system_xyz[2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            dist[dist == 0] = np.nan
            mask = dist > 0.0
            if min_dist > 0:
                mask &= dist >= min_dist
            if max_dist > 0:
                mask &= dist <= max_dist
            absmag = catalog.arrays["absmag"]
            apparent = absmag + 5.0 * np.log10(dist) - 5.0
            mask &= apparent <= mag_limit
            if not mask.any():
                _write_json(self, 200, {"hit": None})
                return
            dx = dx[mask]
            dy = dy[mask]
            dz = dz[mask]
            dist = dist[mask]
            vec = np.stack([dx, dy, dz], axis=1)
            vec /= dist[:, None]
            if payload.get("background_frame") == "galactic":
                vec = _eq_to_gal(vec.T).T
            dot = vec.dot(target)
            idx = int(np.argmax(dot))
            star = catalog.stars[int(np.flatnonzero(mask)[idx])]
            # Calculate calibrated game coordinates
            calibrated = None
            if payload.get("calibration"):
                gx, gy, gz = _hyg_to_game(catalog, payload, (star.x, star.y, star.z))
                calibrated = {"x": gx, "y": gy, "z": gz}
            _write_json(
                self,
                200,
                {
                    "hit": {
                        "name": star.proper or star.bf or star.gl or f"HIP{star.hip}",
                        "hip": star.hip,
                        "ra": star.ra,
                        "dec": star.dec,
                        "x": star.x,
                        "y": star.y,
                        "z": star.z,
                        "mag": star.mag,
                        "spect": star.spect,
                        "dist_pc": star.dist,
                        "system": system_name,
                        "calibrated_xyz": calibrated,
                    }
                },
            )
            return
        if parsed.path == "/api/render":
            payload = _read_json(self)
            catalog_path = payload.get("catalog", "hygdata_v42.csv")
            reload_data = bool(payload.get("reload_data"))
            catalog = _load_catalog_fresh(catalog_path) if reload_data else _load_catalog(catalog_path)
            system_name, system_xyz = _resolve_system_xyz(catalog, payload)
            width = int(payload.get("width", 4096))
            height = int(payload.get("height", width // 2))
            background = payload.get("background")
            format_name = payload.get("format", "png")
            label_entries = _load_label_entries(catalog, payload, system_name)
            const_segments, const_labels, star_labels = _constellation_overlays(catalog, payload)
            img = render_skybox(
                catalog.arrays,
                system_xyz,
                width,
                height,
                float(payload.get("mag_limit", 7.5)),
                background,
                payload.get("background_projection", "equirectangular"),
                payload.get("background_frame", "equatorial"),
                payload.get("star_frame", "equatorial"),
                payload.get("background_fill", "wrap"),
                (
                    float(payload.get("roll", 0.0)),
                    float(payload.get("pitch", 0.0)),
                    float(payload.get("yaw", 0.0)),
                ),
                float(payload.get("splat_sigma", 2.0)),
                float(payload.get("splat_min_sigma", 0.6)),
                not bool(payload.get("use_catalog_mag", False)),
                bool(payload.get("background_destar", False)),
                float(payload.get("destar_percentile", 99.7)),
                float(payload.get("destar_blur", 2.0)),
                int(payload.get("destar_expand", 1)),
                float(payload.get("background_gain", 1.0)),
                float(payload.get("star_gain", 0.2)),
                float(payload.get("exclude_nearby_pc", 0.05)),
                float(payload.get("min_distance_ly", 0.0)) / 3.26156,
                float(payload.get("max_distance_ly", 0.0)) / 3.26156,
                label_entries,
                payload.get("label_color", "#70ff70"),
                int(payload.get("label_font_size", 18)),
                not bool(payload.get("label_no_hip", False)),
                int(payload.get("label_alpha", 200)),
                bool(payload.get("overlay_guides", False)),
                payload.get("guide_color", "#70ff70"),
                int(payload.get("guide_alpha", 140)),
                float(payload.get("guide_meridian_step", 15.0)),
                int(payload.get("guide_label_font_size", 14)),
                float(payload.get("guide_label_lat", 23.5)),
                const_segments,
                const_labels,
                star_labels,
                payload.get("constellation_color", "#70ff70"),
                int(payload.get("constellation_alpha", 220)),
                int(payload.get("constellation_line_width", 2)),
                int(payload.get("constellation_label_font_size", 14)),
                int(payload.get("constellation_star_font_size", 12)),
                int(payload.get("constellation_star_alpha", 160)),
                payload.get("constellation_frame", "equatorial"),
                _build_equator_labels(payload),
                payload.get("equator_label_color", "#ffffff"),
                int(payload.get("equator_label_font_size", 12)),
                int(payload.get("equator_label_alpha", 200)),
            )
            os.makedirs(OUT_DIR, exist_ok=True)
            filename = f"{system_name}_{width}x{height}.{format_name}"
            out_path = os.path.join(OUT_DIR, filename)
            
            img.save(out_path)
            dds_path = None
            game_skymap_copied = False
            if payload.get("convert_dds"):
                out_path = os.path.join("out\\", filename)
                dds_path = _convert_to_dds(
                    out_path,
                    "out_dds",
                    payload.get("dds_format", "BC7_UNORM"),
                    payload.get("texconv_path"),
                )
                # Copy to game skymap path if requested
                game_skymap_path = payload.get("game_skymap_path")
                if dds_path and game_skymap_path:
                    game_skymap_copied = _copy_to_game_skymap(dds_path, game_skymap_path)
            # Convert system_xyz back to game coordinates for client tracking
            game_xyz = None
            if payload.get("calibration"):
                gx, gy, gz = _hyg_to_game(catalog, payload, system_xyz)
                game_xyz = {"x": gx, "y": gy, "z": gz}
            else:
                game_xyz = {"x": system_xyz[0], "y": system_xyz[1], "z": system_xyz[2]}

            _write_json(
                self,
                200,
                {
                    "output": f"/out/{filename}",
                    "dds": f"/out_dds/{os.path.basename(dds_path)}" if dds_path else None,
                    "game_skymap_copied": game_skymap_copied,
                    "system_name": system_name,
                    "system_xyz": game_xyz,
                    "system_hyg": {"x": system_xyz[0], "y": system_xyz[1], "z": system_xyz[2]},
                },
            )
            return
        self.send_error(404)

    def _serve_file(self, path: str) -> None:
        if not os.path.exists(path):
            self.send_error(404)
            return
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def _convert_to_dds(input_path: str, output_dir: str, format_name: str, texconv_path: Optional[str]) -> str:
    import shutil
    import subprocess

    texconv = texconv_path or shutil.which("texconv") or shutil.which("texconv.exe")
    if not texconv:
        raise RuntimeError("texconv not found in PATH; install DirectXTex or add it to PATH.")
    os.makedirs(output_dir, exist_ok=True)
    cmd = [texconv, "-y", "-f", format_name, "-m", "1", "-o", output_dir, input_path]
    subprocess.run(cmd, check=True)
    return os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + ".dds")


def _copy_to_game_skymap(dds_path: str, game_skymap_path: str) -> bool:
    """Copy the generated DDS file to the game skymap location as milkyway_color.dds.

    Returns True if successful, False otherwise.
    """
    import shutil

    try:
        # Expand environment variables in the path (e.g., $WIN_USER)
        expanded_path = os.path.expandvars(game_skymap_path)
        # Ensure the target directory exists
        target_dir = os.path.dirname(expanded_path)
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
        # Copy the file, renaming to milkyway_color.dds
        shutil.copy2(dds_path, expanded_path)
        print(f"Copied DDS to game skymap: {expanded_path}")
        return True
    except Exception as e:
        print(f"Failed to copy DDS to game skymap: {e}")
        return False


def main() -> None:
    port = int(os.environ.get("SKYBOX_PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), SkyboxHandler)
    print(f"Skybox server listening on http://localhost:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
