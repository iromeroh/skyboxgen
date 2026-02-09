import csv

import numpy as np

from skyboxgen.catalog import apply_offset, compute_offset, load_hyg_catalog


def _write_hyg(path):
    header = [
        "id",
        "hip",
        "hd",
        "hr",
        "gl",
        "bf",
        "proper",
        "ra",
        "dec",
        "dist",
        "pmra",
        "pmdec",
        "rv",
        "mag",
        "absmag",
        "spect",
        "ci",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "rarad",
        "decrad",
        "pmrarad",
        "pmdecrad",
        "bayer",
        "flam",
        "con",
    ]
    rows = [
        {
            "id": "1",
            "hip": "123",
            "proper": "Alpha Centauri",
            "ra": "0",
            "dec": "0",
            "dist": "1.34",
            "mag": "0.0",
            "absmag": "0.0",
            "spect": "G2V",
            "ci": "0.65",
            "x": "0.5",
            "y": "0.25",
            "z": "-1.0",
        },
        {
            "id": "2",
            "hip": "456",
            "proper": "Beta",
            "ra": "0",
            "dec": "0",
            "dist": "2.0",
            "mag": "1.0",
            "absmag": "0.0",
            "spect": "K0V",
            "ci": "0.8",
            "x": "2.0",
            "y": "0.0",
            "z": "0.0",
        },
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_catalog_offset(tmp_path):
    csv_path = tmp_path / "hyg.csv"
    _write_hyg(csv_path)
    catalog = load_hyg_catalog(str(csv_path))
    offset = compute_offset(catalog, "Alpha Centauri", (-0.5, -0.5, -1.5))
    apply_offset(catalog, offset)
    assert np.isclose(catalog.stars[0].x, -0.5)
    assert np.isclose(catalog.stars[0].y, -0.5)
    assert np.isclose(catalog.stars[0].z, -1.5)
