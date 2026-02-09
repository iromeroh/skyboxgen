import csv
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class StarfieldSystem:
    name: str
    faction: str
    bodies: str
    inorg_res: str
    org_res: str
    flora: str
    fauna: str
    level: str
    x: Optional[float]
    y: Optional[float]
    z: Optional[float]
    fictional: Optional[bool]


def load_starfield_systems(path: str) -> List[StarfieldSystem]:
    # Parse the mixed comma-header + tab-body format into structured system rows.
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    if not lines:
        return []
    raw_header = [h.strip() for h in lines[0].split(",")]
    header = [h.lower() for h in raw_header]
    systems = []
    for line in lines[1:]:
        if "\t" in line:
            parts = [p.strip() for p in line.split("\t")]
        else:
            parts = [p.strip() for p in line.split(",")]
        while len(parts) < len(header):
            parts.append("")
        row = dict(zip(header, parts))
        # If x/y/z are appended without headers, read them by position.
        x = _safe_float(row.get("x", ""))
        y = _safe_float(row.get("y", ""))
        z = _safe_float(row.get("z", ""))
        if x is None and len(parts) >= len(header) + 3:
            x = _safe_float(parts[len(header)])
            y = _safe_float(parts[len(header) + 1])
            z = _safe_float(parts[len(header) + 2])
        systems.append(
            StarfieldSystem(
                name=row.get("star_system", "").strip(),
                faction=row.get("faction", "").strip(),
                bodies=row.get("bodies", "").strip(),
                inorg_res=row.get("inorg_res", "").strip(),
                org_res=row.get("org_res", "").strip(),
                flora=row.get("flora", "").strip(),
                fauna=row.get("fauna", "").strip(),
                level=row.get("level", "").strip(),
                x=x,
                y=y,
                z=z,
                fictional=_safe_bool(row.get("fictional", "")),
            )
        )
    return systems


def write_game_stars_csv(
    systems: List[StarfieldSystem],
    out_path: str,
    mark_fictional_with_xyz: bool = True,
) -> None:
    # Emit a standardized CSV template for the generator, keeping Starfield metadata implicit.
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "game_name",
                "hyg_query",
                "hip",
                "ra_deg",
                "dec_deg",
                "dist_pc",
                "x",
                "y",
                "z",
                "fictional",
            ]
        )
        for system in systems:
            name = system.name
            if not name:
                continue
            has_xyz = system.x is not None or system.y is not None or system.z is not None
            if system.fictional is not None:
                fictional = "true" if system.fictional else "false"
            else:
                fictional = "true" if mark_fictional_with_xyz and has_xyz else "false"
            writer.writerow(
                [
                    name,
                    name,
                    "",
                    "",
                    "",
                    "",
                    system.x if system.x is not None else "",
                    system.y if system.y is not None else "",
                    system.z if system.z is not None else "",
                    fictional,
                ]
            )


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value) if value != "" else None
    except ValueError:
        return None


def _safe_bool(value: str) -> Optional[bool]:
    if value == "":
        return None
    val = value.strip().lower()
    if val in ("true", "yes", "1"):
        return True
    if val in ("false", "no", "0"):
        return False
    return None
