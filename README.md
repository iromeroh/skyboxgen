# Starfield Skybox Generator (HYG)

Generates equirectangular 4K skybox textures for Starfield systems using HYG star data, with optional 2MASS Milky Way background reprojection and caption output for named stars.

## Setup

```
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Interactive mode

```
python3 -m skyboxgen.cli interactive --catalog hygdata_v42.csv \
  --background Allsky-Stars_full.jpg --background-projection mollweide \
  --background-fill wrap --background-gain 1.0 \
  --background-destar \
  --label-game-stars data/game_stars.csv --overlay-guides --guide-meridian-step 15 \
  --overlay-constellations --constellation-star-labels \
  --mag-limit 7.5 --format png --caption
```

## Earth centered 1000 year away equatorial map with constellations and lines
python3 -m skyboxgen.cli interactive --system-name "Sol"  --system-x 0 --system-y 0 --system-z 0 \
--system-frame equatorial --background backgrounds/diffuse_background.jpg --background-projection equirectangular \
--label-game-stars data/game_stars.csv --overlay-guides --label-game-stars data/game_stars.csv  \
--overlay-constellations --constellation-star-labels  --constellation-color 1111ff --size 8192 \
--convert-dds  --texconv-path ~/.local/bin/texconv.exe --min-distance-ly 1000 --mag-limit 9.5 \
--constellation-frame equatorial  --background-frame equatorial

Fictional system via flags:

```
python3 -m skyboxgen.cli interactive --system-name "Kryx" \
  --system-x 0 --system-y 0 --system-z 0 --system-frame equatorial --format png
```

Fictional system via equatorial coordinates:

```
python3 -m skyboxgen.cli interactive --system-name "Kryx" \
  --system-ra 120.5 --system-dec -23.2 --system-dist 120 \
  --format png
```

With equator labels (3 copies of the system name at the equator for in-game visibility):

```
python3 -m skyboxgen.cli interactive --system-name "Sol" \
  --equator-labels "Sol" --equator-label-color "#ffffff" \
  --overlay-guides --format png
```

## Batch mode

```
python3 -m skyboxgen.cli batch --catalog hygdata_v42.csv \
  --background Allsky-Stars_full.jpg --background-projection mollweide \
  --background-fill wrap --background-gain 1.0 \
  --background-destar \
  --mag-limit 7.5 --systems named
```

Use `--systems all` for every HYG entry (slow and large).

## Captions

Create a text file with star names (one per line), then:

```
python3 -m skyboxgen.cli captions --names named_stars.txt --out captions.csv
```

## Tests

```
pytest
```

## Viewer

Run the local server, then open the viewer in a browser:

```
python3 skybox_server.py
```

Then open `http://localhost:8000/`.

The web viewer supports all rendering options including equator labels (enter text in the "Equator labels (3x)" field), guide overlays, constellation lines, and DDS conversion.

## Background preprocessing

Create a reusable equirectangular Milky Way backdrop (with optional de-star and border cleanup):

```
python3 -m skyboxgen.cli preprocess-background \
  --input Allsky-Stars_full.jpg --output backgrounds/2mass_equirect.png \
  --size 4096 --projection mollweide --fill wrap \
  --destar --destar-percentile 99.95 --destar-blur 1.0 --destar-expand 0 \
  --border-mask --border-tolerance 0.05 --gain 1.4
```

Use the saved output in future renders with `--background-projection equirectangular`.

To compute apparent constellations from a specific system:

```
python3 -m skyboxgen.cli captions --names named_stars.txt \
  --system-x 10 --system-y 5 --system-z -2 \
  --system-frame equatorial \
  --constellations constbnd/conbound.c \
  --out captions.csv
```

## Game star list

Edit `data/game_stars.csv` with your Starfield systems and optional fictional coordinates (assumed equatorial XYZ unless noted), then:

```
python3 -m skyboxgen.cli batch --game-stars data/game_stars.csv \
  --background Allsky-Stars_full.jpg --background-projection mollweide \
  --background-destar --mag-limit 7.5
```

Convert the Starfield system list to the game stars CSV (optional `x,y,z,fictional` columns supported):

```
python3 -m skyboxgen.cli import-starfield --input starfield_systems.txt --out data/game_stars.csv
```

If you add `x,y,z` columns (parsecs, Sol-centered) to `starfield_systems.txt`, the importer will carry them into `data/game_stars.csv` and mark those rows as fictional. You can append `x,y,z` after the existing columns even if the header doesn’t include them. Use `--no-mark-fictional` to keep `fictional=false` even when `x,y,z` are present.

Suggest HYG matches for missing game star names:

```
python3 -m skyboxgen.cli match-game-stars --game-stars data/game_stars.csv --out data/game_stars_matches.csv
```

Suggest HYG matches using Starfield parsec coordinates from `starfield_systems.txt`:

```
python3 -m skyboxgen.cli match-starfield --input starfield_systems.txt --out data/starfield_matches.csv
```

Apply best matches into `data/game_stars.csv` (adds `hyg_match`, fills `hip`/`hyg_query` if empty):

```
python3 -m skyboxgen.cli match-starfield --input starfield_systems.txt \
  --apply --game-stars data/game_stars.csv --apply-out data/game_stars_with_matches.csv
```

When `x,y,z` are present and a matching HYG star is found, the importer also writes `data/game_stars_coord_deltas.csv` showing the coordinate difference after calibration.

Add nearby real-star suggestions (within 1 ly):

```
python3 -m skyboxgen.cli import-starfield --input starfield_systems.txt \
  --out data/game_stars.csv --suggest-nearby --suggest-distance-ly 1.0
```

## Notes

- HYG coordinates are equatorial (x/y/z). Use `--background-frame galactic` (default) to align with Milky Way backgrounds, or set both to `equatorial` for non-galactic backgrounds.
- Calibration aligns the HYG catalog with in-game coordinates using Alpha Centauri by default; disable with `--no-calibration`.
- RA/Dec inputs are converted to equatorial XYZ and then shifted by the same calibration offset so they align with in-game positions.
- Background projection is assumed to be Mollweide for 2MASS; set `--background-projection equirectangular` for already-projected maps.
- `--background-fill wrap` removes the ellipse border by wrapping the Mollweide map across the full equirectangular output.
- `--background-gain` scales the background brightness to make the Milky Way band more visible.
- `--star-gain` scales star splat brightness; lower it if the field looks saturated.
- `--exclude-nearby-pc` drops stars closer than the given parsec distance (useful to hide the system’s primary star).
- `--min-distance-ly` and `--max-distance-ly` limit which stars render by distance from the current system (in light years).
- De-star filtering is controlled via `--background-destar` plus `--destar-percentile`, `--destar-blur`, and `--destar-expand`.
- Use `--label-game-stars data/game_stars.csv` to label Starfield systems in the skybox, and `--overlay-guides` for galactic equator/tropics and meridians.
- Guide labels can be tuned with `--guide-label-font-size` and `--guide-label-lat` (default 23.5° at the tropics for better readability); meridian spacing with `--guide-meridian-step`.
- Use `--overlay-constellations` with `data/ConstellationLines.csv` to draw constellation lines. Defaults are boosted for visibility; tune with `--constellation-alpha` and `--constellation-line-width`.
- Constellation overlays default to the equatorial frame (`--constellation-frame equatorial`) so Sol matches familiar sky maps.
- For best alignment between star field and constellation lines, set `--background-frame` and `--constellation-frame` to the same value.
- Use `--equator-labels "Name"` to place 3 copies of a label at the equator, spaced 120° apart, ensuring visibility from any in-game viewing angle. Customize with `--equator-label-color` (default white), `--equator-label-font-size`, and `--equator-label-alpha`.
- Use `--convert-dds` to run `texconv` after rendering; output goes to `out_dds` and uses `BC7_UNORM` by default. If `texconv.exe` is on PATH but not detected, pass `--texconv-path` explicitly.
- Constellation lookup uses the B1875 boundary data from `constbnd/conbound.c` without precession.
- DDS output depends on your Pillow build. If DDS save fails, convert PNGs using an external tool like `texconv`.
