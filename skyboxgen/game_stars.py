import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .catalog import Catalog, Star, search_stars
from .coords import radec_to_equatorial_xyz


@dataclass
class GameStar:
    game_name: str
    hyg_query: str
    hip: Optional[int]
    ra_deg: Optional[float]
    dec_deg: Optional[float]
    dist_pc: Optional[float]
    x: Optional[float]
    y: Optional[float]
    z: Optional[float]
    fictional: bool


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value) if value != "" else None
    except ValueError:
        return None


def _safe_int(value: str) -> Optional[int]:
    try:
        return int(value) if value != "" else None
    except ValueError:
        return None


def load_game_stars(path: str) -> List[GameStar]:
    stars: List[GameStar] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in reader:
            stars.append(
                GameStar(
                    game_name=row.get("game_name", "").strip(),
                    hyg_query=row.get("hyg_query", "").strip(),
                    hip=_safe_int(row.get("hip", "")),
                    ra_deg=_safe_float(row.get("ra_deg", "")),
                    dec_deg=_safe_float(row.get("dec_deg", "")),
                    dist_pc=_safe_float(row.get("dist_pc", "")),
                    x=_safe_float(row.get("x", "")),
                    y=_safe_float(row.get("y", "")),
                    z=_safe_float(row.get("z", "")),
                    fictional=(row.get("fictional", "").strip().lower() == "true"),
                )
            )
    return stars


def build_name_index(catalog: Catalog) -> Dict[str, List[Star]]:
    # Build a lowercase index across proper, Bayer/Flamsteed, and GL names.
    index: Dict[str, List[Star]] = {}
    for key, stars in catalog.by_name.items():
        index[key] = stars
    return index


def resolve_game_star(
    catalog: Catalog, entry: GameStar
) -> Tuple[str, Tuple[float, float, float], Optional[Star]]:
    if entry.fictional:
        if entry.x is not None and entry.y is not None and entry.z is not None:
            return entry.game_name, (entry.x, entry.y, entry.z), None
        if entry.ra_deg is not None and entry.dec_deg is not None and entry.dist_pc is not None:
            return entry.game_name, radec_to_equatorial_xyz(entry.ra_deg, entry.dec_deg, entry.dist_pc), None
        raise ValueError(f"Fictional system missing coordinates: {entry.game_name}")
    if entry.hip is not None:
        matches = search_stars(catalog, str(entry.hip), limit=1)
        if matches:
            star = matches[0]
            return entry.game_name or star.proper or star.bf or star.gl, (star.x, star.y, star.z), star
    if entry.hyg_query:
        matches = search_stars(catalog, entry.hyg_query, limit=1)
        if matches:
            star = matches[0]
            return entry.game_name or star.proper or star.bf or star.gl, (star.x, star.y, star.z), star
    raise ValueError(f"Unable to resolve game star: {entry.game_name}")
