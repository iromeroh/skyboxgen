from typing import Iterable, List, Optional, Tuple

from .catalog import Catalog, Star, search_stars
from .constellations import ConstellationTable, constell_from_ra_dec, load_constellation_table
from .coords import vector_to_radec


def write_caption(
    star: Star,
    path: str,
    system_name: str,
    apparent_constellation: Optional[str] = None,
) -> None:
    # Keep the text short, dialog-oriented, and rich with astronomy details for TTS.
    distance_ly = star.dist * 3.26156
    primary_name = star.proper or star.bf or star.gl or f"HIP {star.hip}" if star.hip else system_name
    lines = [
        f"System data slate for {system_name}.",
        f"This is {primary_name}. HIP {star.hip or 'unknown'}.",
        f"Equatorial coordinates: RA {star.ra:.3f} degrees, Dec {star.dec:.3f} degrees.",
        f"Distance from Sol: {star.dist:.3f} parsecs, about {distance_ly:.2f} light years.",
        f"Apparent magnitude from Earth: {star.mag:.2f}. Absolute magnitude: {star.absmag:.2f}.",
        f"Spectral type: {star.spect or 'n/a'}. Color index B minus V: {star.ci:.2f}.",
        f"Constellation from Earth: {star.con or 'n/a'}.",
    ]
    if apparent_constellation:
        lines.append(f"From this system, it appears in {apparent_constellation}.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_captions_for_names(
    catalog: Catalog,
    names: Iterable[str],
    path: str,
    system_xyz: Optional[Tuple[float, float, float]] = None,
    conbound_c_path: Optional[str] = None,
) -> List[str]:
    entries = []
    missing = []
    table = None
    if system_xyz and conbound_c_path:
        table = load_constellation_table(conbound_c_path)
    for name in names:
        matches = search_stars(catalog, name, limit=1)
        if not matches:
            missing.append(name)
            continue
        star = matches[0]
        apparent_const = ""
        if table and system_xyz:
            apparent_const = apparent_constellation_for_star(table, system_xyz, star)
        entries.append(
            [
                star.proper or star.bf or star.gl or name,
                str(star.hip or ""),
                f"{star.ra:.6f}",
                f"{star.dec:.6f}",
                f"{star.dist:.3f}",
                f"{star.mag:.2f}",
                star.spect or "",
                star.con or "",
                apparent_const,
            ]
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("name,hip,ra_deg,dec_deg,dist_pc,mag,spect,con,apparent_con\n")
        for row in entries:
            f.write(",".join(row) + "\n")
        if missing:
            f.write("# missing: " + ", ".join(missing) + "\n")
    return missing


def read_name_list(path: str) -> List[str]:
    names: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            names.append(line)
    return names


def apparent_radec(
    system_xyz: Tuple[float, float, float], star_xyz: Tuple[float, float, float]
) -> Tuple[float, float]:
    dx = star_xyz[0] - system_xyz[0]
    dy = star_xyz[1] - system_xyz[1]
    dz = star_xyz[2] - system_xyz[2]
    return vector_to_radec((dx, dy, dz))


def apparent_constellation_for_star(
    table: ConstellationTable, system_xyz: Tuple[float, float, float], star: Star
) -> str:
    ra_deg, dec_deg = apparent_radec(system_xyz, (star.x, star.y, star.z))
    return constell_from_ra_dec(table, ra_deg, dec_deg)
