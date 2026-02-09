import math
from typing import Tuple

import numpy as np


_EQ_TO_GAL = np.array(
    [
        [-0.0548755604, -0.8734370902, -0.4838350155],
        [0.4941094279, -0.4448296300, 0.7469822445],
        [-0.8676661490, -0.1980763734, 0.4559837762],
    ],
    dtype=np.float64,
)

_GAL_TO_EQ = np.linalg.inv(_EQ_TO_GAL)


def radec_to_equatorial_xyz(ra_deg: float, dec_deg: float, dist: float) -> Tuple[float, float, float]:
    ra = math.radians(ra_deg)
    dec = math.radians(dec_deg)
    x = dist * math.cos(dec) * math.cos(ra)
    y = dist * math.cos(dec) * math.sin(ra)
    z = dist * math.sin(dec)
    return x, y, z


def radec_to_galactic_xyz(ra_deg: float, dec_deg: float, dist: float) -> Tuple[float, float, float]:
    vec = np.array(radec_to_equatorial_xyz(ra_deg, dec_deg, dist))
    vec = np.dot(_EQ_TO_GAL, vec)
    return float(vec[0]), float(vec[1]), float(vec[2])


def equatorial_to_galactic(vec: np.ndarray) -> np.ndarray:
    return np.dot(_EQ_TO_GAL, vec)


def galactic_to_equatorial(vec: np.ndarray) -> np.ndarray:
    return np.dot(_GAL_TO_EQ, vec)


def vector_to_radec(vec: Tuple[float, float, float]) -> Tuple[float, float]:
    x, y, z = vec
    r = math.sqrt(x * x + y * y + z * z)
    if r == 0:
        return 0.0, 0.0
    ra = math.degrees(math.atan2(y, x)) % 360.0
    dec = math.degrees(math.asin(z / r))
    return ra, dec
