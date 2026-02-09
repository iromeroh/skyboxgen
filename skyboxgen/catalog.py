import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Star:
    idx: int
    hip: Optional[int]
    proper: str
    bf: str
    gl: str
    hd: str
    hr: str
    ra: float
    dec: float
    dist: float
    mag: float
    absmag: float
    spect: str
    ci: float
    x: float
    y: float
    z: float
    con: str


class Catalog:
    def __init__(self, stars: List[Star], arrays: Dict[str, np.ndarray]) -> None:
        # Keep both list and vectorized views so rendering stays fast.
        self.stars = stars
        self.arrays = arrays
        self.by_hip = {s.hip: s for s in stars if s.hip is not None}
        self.by_hr = {}
        self.by_name = {}
        for s in stars:
            if s.hr.isdigit():
                hr = int(s.hr)
                if hr not in self.by_hr:
                    self.by_hr[hr] = s
            for name in [s.proper, s.bf, s.gl]:
                if name:
                    key = name.lower()
                    self.by_name.setdefault(key, []).append(s)
                    norm = _normalize_name(key)
                    if norm != key:
                        self.by_name.setdefault(norm, []).append(s)


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: str) -> Optional[int]:
    try:
        if value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _compute_absmag(mag: float, dist: float, absmag: float) -> float:
    if absmag != 0.0:
        return absmag
    if dist <= 0.0:
        return mag
    return mag - 5.0 * (np.log10(dist) - 1.0)


def load_hyg_catalog(path: str) -> Catalog:
    # HYG is CSV with equatorial x/y/z in parsecs; we normalize into numeric arrays.
    stars: List[Star] = []
    arrays = {
        "x": [],
        "y": [],
        "z": [],
        "absmag": [],
        "mag": [],
        "ci": [],
    }
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            hip = _safe_int(row.get("hip", ""))
            proper = (row.get("proper", "") or "").strip()
            bf = (row.get("bf", "") or "").strip()
            gl = (row.get("gl", "") or "").strip()
            hd = (row.get("hd", "") or "").strip()
            hr = (row.get("hr", "") or "").strip()
            ra = _safe_float(row.get("ra", "0"))
            dec = _safe_float(row.get("dec", "0"))
            dist = _safe_float(row.get("dist", "0"))
            mag = _safe_float(row.get("mag", "0"))
            absmag = _safe_float(row.get("absmag", "0"))
            spect = (row.get("spect", "") or "").strip()
            ci = _safe_float(row.get("ci", "0"))
            x = _safe_float(row.get("x", "0"))
            y = _safe_float(row.get("y", "0"))
            z = _safe_float(row.get("z", "0"))
            con = (row.get("con", "") or "").strip()
            absmag = _compute_absmag(mag, dist, absmag)
            star = Star(
                idx=idx,
                hip=hip,
                proper=proper,
                bf=bf,
                gl=gl,
                hd=hd,
                hr=hr,
                ra=ra,
                dec=dec,
                dist=dist,
                mag=mag,
                absmag=absmag,
                spect=spect,
                ci=ci,
                x=x,
                y=y,
                z=z,
                con=con,
            )
            stars.append(star)
            arrays["x"].append(x)
            arrays["y"].append(y)
            arrays["z"].append(z)
            arrays["absmag"].append(absmag)
            arrays["mag"].append(mag)
            arrays["ci"].append(ci)
    for k, v in arrays.items():
        arrays[k] = np.asarray(v, dtype=np.float32)
    return Catalog(stars, arrays)


def apply_offset(catalog: Catalog, offset: Tuple[float, float, float]) -> None:
    # Translate all coordinates to align the catalog with an in-game reference frame.
    ox, oy, oz = offset
    for star in catalog.stars:
        star.x += ox
        star.y += oy
        star.z += oz
    catalog.arrays["x"] = (catalog.arrays["x"] + ox).astype(np.float32)
    catalog.arrays["y"] = (catalog.arrays["y"] + oy).astype(np.float32)
    catalog.arrays["z"] = (catalog.arrays["z"] + oz).astype(np.float32)


def compute_offset(
    catalog: Catalog, name_or_hip: str, target_xyz: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    # Compute a simple translation so a known star matches a known in-game coordinate.
    # Accept common aliases to smooth over naming differences in the HYG catalog.
    query = name_or_hip.strip()
    if query.lower().startswith("hip"):
        query = query[3:]
    matches = search_stars(catalog, query, limit=1)
    if not matches and query.lower() in _CALIBRATION_ALIASES:
        for alias in _CALIBRATION_ALIASES[query.lower()]:
            matches = search_stars(catalog, alias, limit=1)
            if matches:
                break
    if not matches:
        raise ValueError(f"Calibration star not found: {name_or_hip}")
    star = matches[0]
    tx, ty, tz = target_xyz
    return (tx - star.x, ty - star.y, tz - star.z)


_CALIBRATION_ALIASES = {
    "alpha centauri": ["Rigil Kentaurus", "Toliman", "Alp1Cen", "Alp2Cen", "71683"],
}


def search_stars(catalog: Catalog, query: str, limit: int = 10) -> List[Star]:
    q = query.strip().lower()
    if not q:
        return []
    if q.isdigit():
        hip = int(q)
        if hip in catalog.by_hip:
            return [catalog.by_hip[hip]]
    direct = catalog.by_name.get(q, [])
    if direct:
        return direct[:limit]
    q_norm = _normalize_name(q)
    if q_norm != q:
        direct = catalog.by_name.get(q_norm, [])
        if direct:
            return direct[:limit]
    matches = []
    for name, stars in catalog.by_name.items():
        if q in name:
            matches.extend(stars)
            if len(matches) >= limit:
                break
    return matches[:limit]


def resolve_system(
    catalog: Catalog,
    name: Optional[str],
    hip: Optional[int],
    system_xyz: Optional[Tuple[float, float, float]],
) -> Tuple[str, Tuple[float, float, float], Optional[Star]]:
    if system_xyz is not None:
        system_name = name or "FictionalSystem"
        return system_name, system_xyz, None
    if hip is not None and hip in catalog.by_hip:
        star = catalog.by_hip[hip]
        system_name = name or star.proper or star.bf or star.gl or f"HIP{star.hip}"
        return system_name, (star.x, star.y, star.z), star
    if name:
        matches = search_stars(catalog, name, limit=1)
        if matches:
            star = matches[0]
            system_name = star.proper or star.bf or star.gl or f"HIP{star.hip}"
            return system_name, (star.x, star.y, star.z), star
    raise ValueError("Unable to resolve system from provided inputs.")


def _normalize_name(name: str) -> str:
    # Normalize common catalog aliases like Gliese/GJ identifiers.
    key = name.replace(" ", "")
    key = key.replace("gliese", "gl")
    key = key.replace("gj", "gj")
    if key.startswith("gl"):
        key = "gl" + key[2:]
    if key.startswith("gj"):
        key = "gj" + key[2:]
    return key
