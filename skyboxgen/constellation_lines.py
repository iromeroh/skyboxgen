import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .catalog import Catalog, Star


@dataclass
class ConstellationLine:
    abr: str
    hr_sequence: List[int]


def load_constellation_lines(path: str) -> List[ConstellationLine]:
    # Parse Marc vd Sluys' HR-number line list into ordered sequences.
    lines: List[ConstellationLine] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, skipinitialspace=True)
        header = next(reader, None)
        if not header:
            return lines
        for row in reader:
            if not row or not row[0].strip():
                continue
            abr = row[0].strip()
            try:
                count = int(row[1].strip())
            except (IndexError, ValueError):
                continue
            hr_numbers: List[int] = []
            for cell in row[2 : 2 + count]:
                cell = cell.strip()
                if not cell:
                    continue
                try:
                    hr_numbers.append(int(cell))
                except ValueError:
                    continue
            if len(hr_numbers) >= 2:
                lines.append(ConstellationLine(abr=abr, hr_sequence=hr_numbers))
    return lines


def build_constellation_overlays(
    catalog: Catalog,
    line_path: str,
    star_mag_limit: float,
) -> Tuple[List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]], List[Tuple[str, Tuple[float, float, float]]], List[Tuple[str, Tuple[float, float, float]]]]:
    # Convert HR sequences into 3D line segments and label positions.
    lines = load_constellation_lines(line_path)
    segments: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = []
    const_labels: List[Tuple[str, Tuple[float, float, float]]] = []
    star_labels: List[Tuple[str, Tuple[float, float, float]]] = []

    for line in lines:
        stars: List[Star] = []
        for hr in line.hr_sequence:
            star = catalog.by_hr.get(hr)
            if star:
                stars.append(star)
        if len(stars) < 2:
            continue
        for a, b in zip(stars[:-1], stars[1:]):
            segments.append(((a.x, a.y, a.z), (b.x, b.y, b.z)))
        center = _mean_position(stars)
        if center:
            const_labels.append((line.abr, center))
        for star in stars:
            if star.mag <= star_mag_limit:
                name = star.proper or star.bf or star.gl
                if name:
                    star_labels.append((name, (star.x, star.y, star.z)))
    return segments, const_labels, star_labels


def _mean_position(stars: List[Star]) -> Optional[Tuple[float, float, float]]:
    if not stars:
        return None
    x = sum(s.x for s in stars) / len(stars)
    y = sum(s.y for s in stars) / len(stars)
    z = sum(s.z for s in stars) / len(stars)
    return (x, y, z)
