import math
from typing import Tuple


def bv_to_rgb(bv: float) -> Tuple[float, float, float]:
    # Approximate color from B-V index.
    bv = max(-0.4, min(2.0, bv))
    t = 4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))
    return _temp_to_rgb(t)


def _temp_to_rgb(kelvin: float) -> Tuple[float, float, float]:
    temp = kelvin / 100.0
    if temp <= 66:
        r = 255
        g = 99.4708025861 * math.log(temp) - 161.1195681661
        b = 0 if temp <= 19 else 138.5177312231 * math.log(temp - 10) - 305.0447927307
    else:
        r = 329.698727446 * math.pow(temp - 60, -0.1332047592)
        g = 288.1221695283 * math.pow(temp - 60, -0.0755148492)
        b = 255
    return (
        max(0.0, min(1.0, r / 255.0)),
        max(0.0, min(1.0, g / 255.0)),
        max(0.0, min(1.0, b / 255.0)),
    )
