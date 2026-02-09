import re
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Bound:
    ra_spd: int
    d_ra: int
    constell_idx: int


@dataclass
class ConstellationTable:
    names: str
    bounds: List[Bound]


_BOUND_RE = re.compile(r"\{\s*(0x[0-9a-fA-F]+)\s*,\s*(0x[0-9a-fA-F]+)\s*,\s*(\d+)\s*\}")


def load_constellation_table(conbound_c_path: str) -> ConstellationTable:
    with open(conbound_c_path, "r", encoding="utf-8") as f:
        data = f.read()
    names = _extract_constellation_names(data)
    if not names:
        raise ValueError("Unable to parse constellation names from conbound.c")
    bounds = []
    for ra_spd_hex, d_ra_hex, idx_str in _BOUND_RE.findall(data):
        bounds.append(Bound(int(ra_spd_hex, 16), int(d_ra_hex, 16), int(idx_str)))
    if not bounds:
        raise ValueError("Unable to parse constellation bounds from conbound.c")
    return ConstellationTable(names=names, bounds=bounds)


def _extract_constellation_names(data: str) -> str:
    # Parse the quoted name blocks following the constell_names declaration.
    marker = "const char *constell_names"
    start = data.find(marker)
    if start == -1:
        return ""
    after = data.find("=", start)
    if after == -1:
        return ""
    end = data.find(";", after)
    if end == -1:
        return ""
    block = data[after + 1 : end]
    return "".join(re.findall(r'"([^"]+)"', block))


def constell_from_ra_dec(
    table: ConstellationTable, ra_deg: float, dec_deg: float
) -> str:
    ra = int(ra_deg * 240.0) % 86400
    spd = int((dec_deg + 90.0) * 60.0)
    idx = -1
    step = 512
    while step >> 1:
        step >>= 1
        if idx + step < len(table.bounds) and spd < _spd(table.bounds[idx + step]):
            idx += step
    rval = -1
    while idx >= 0 and rval == -1:
        min_ra = _ra(table.bounds[idx])
        max_ra = min_ra + table.bounds[idx].d_ra
        if (ra >= min_ra and ra < max_ra) or (ra + 86400 >= min_ra and ra + 86400 < max_ra):
            rval = table.bounds[idx].constell_idx
        idx -= 1
    if rval == -1:
        rval = 83  # UMi fallback
    return table.names[3 * rval : 3 * rval + 3]


def _spd(bound: Bound) -> int:
    return bound.ra_spd >> 17


def _ra(bound: Bound) -> int:
    return bound.ra_spd & 0x1FFFF
