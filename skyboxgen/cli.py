import argparse
import os
import difflib
import shutil
import subprocess
from typing import Optional, Tuple

import numpy as np
from PIL import Image

from .background import preprocess_background
from .captions import (
    apparent_constellation_for_star,
    read_name_list,
    write_caption,
    write_captions_for_names,
)
from .constellations import load_constellation_table
from .catalog import apply_offset, compute_offset, load_hyg_catalog, search_stars
from .coords import galactic_to_equatorial, radec_to_equatorial_xyz
from .constellation_lines import build_constellation_overlays
from .game_stars import build_name_index, load_game_stars, resolve_game_star
from .render import render_skybox
from .starfield import load_starfield_systems, write_game_stars_csv


def _parse_system_xyz(
    args: argparse.Namespace, offset: Optional[Tuple[float, float, float]] = None
) -> Optional[Tuple[float, float, float]]:
    # Interpret system positions as equatorial XYZ by default for consistency with HYG.
    if args.system_x is not None and args.system_y is not None and args.system_z is not None:
        system = (args.system_x, args.system_y, args.system_z)
        if args.system_frame == "galactic":
            vec = galactic_to_equatorial(
                np.array([system[0], system[1], system[2]], dtype=float)
            )
            return (float(vec[0]), float(vec[1]), float(vec[2]))
        return system
    if args.system_ra is not None and args.system_dec is not None and args.system_dist is not None:
        # RA/Dec are converted to equatorial XYZ and then shifted into the calibrated frame.
        x, y, z = radec_to_equatorial_xyz(args.system_ra, args.system_dec, args.system_dist)
        if offset:
            x += offset[0]
            y += offset[1]
            z += offset[2]
        return (x, y, z)
    return None


def _sanitize_name(name: str) -> str:
    keep = [c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip()]
    return "".join(keep)[:64] or "system"


def _save_image(img: Image.Image, out_path: str, fmt: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if fmt.lower() == "dds":
        try:
            img.save(out_path)
        except Exception as exc:  # Pillow may not support DDS in this build
            raise RuntimeError("DDS save failed; consider converting PNG with texconv.") from exc
    else:
        img.save(out_path, format=fmt.upper())


def _convert_to_dds(
    input_path: str,
    output_dir: str,
    format_name: str,
    texconv_path: Optional[str] = None,
) -> str:
    # Use texconv when available to generate Starfield-compatible DDS output.
    texconv = texconv_path or shutil.which("texconv") or shutil.which("texconv.exe")
    if not texconv:
        raise RuntimeError("texconv not found in PATH; install DirectXTex or add it to PATH.")
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        texconv,
        "-f",
        format_name,
        "-m",
        "1",
        "-o",
        output_dir,
        input_path,
    ]
    subprocess.run(cmd, check=True)
    return os.path.join(output_dir, os.path.splitext(os.path.basename(input_path))[0] + ".dds")


def run_interactive(args: argparse.Namespace) -> None:
    catalog, offset = _load_catalog_with_calibration(args)
    system_xyz = _parse_system_xyz(args, offset=offset)
    system_star = None
    if system_xyz is not None:
        system_name = args.system_name or "FictionalSystem"
    elif args.system_hip is not None:
        matches = search_stars(catalog, str(args.system_hip), limit=1)
        if not matches:
            raise SystemExit("No matching system found for provided HIP.")
        system_star = matches[0]
        system_name = system_star.proper or system_star.bf or system_star.gl or f"HIP{system_star.hip}"
        system_xyz = (system_star.x, system_star.y, system_star.z)
    elif args.system_name:
        matches = search_stars(catalog, args.system_name, limit=5)
        if not matches:
            raise SystemExit("No matching system found. Try a different name or HIP ID.")
        if len(matches) > 1:
            for i, star in enumerate(matches, 1):
                label = star.proper or star.bf or star.gl or f"HIP{star.hip}"
                print(f"{i}. {label} (HIP {star.hip or 'n/a'})")
            pick = int(input("Pick system number: ").strip())
            system_star = matches[pick - 1]
        else:
            system_star = matches[0]
        system_name = system_star.proper or system_star.bf or system_star.gl or f"HIP{system_star.hip}"
        system_xyz = (system_star.x, system_star.y, system_star.z)
    else:
        query = input("Enter system name or HIP ID (or type 'fictional'): ").strip()
        if query.lower() == "fictional":
            system_name = input("System name: ").strip() or "FictionalSystem"
            frame = input("Coordinate frame (galactic/equatorial) [galactic]: ").strip().lower() or "galactic"
            if frame == "equatorial":
                ra = float(input("RA degrees: ").strip())
                dec = float(input("DEC degrees: ").strip())
                dist = float(input("Distance (pc): ").strip())
                x, y, z = radec_to_equatorial_xyz(ra, dec, dist)
                system_xyz = (x + offset[0], y + offset[1], z + offset[2])
            else:
                x = float(input("X (pc): ").strip())
                y = float(input("Y (pc): ").strip())
                z = float(input("Z (pc): ").strip())
                vec = galactic_to_equatorial(np.array([x, y, z], dtype=float))
                system_xyz = (
                    float(vec[0]) + offset[0],
                    float(vec[1]) + offset[1],
                    float(vec[2]) + offset[2],
                )
        else:
            matches = search_stars(catalog, query, limit=5)
            if not matches:
                raise SystemExit("No matching system found. Try a different name or HIP ID.")
            if len(matches) > 1:
                for i, star in enumerate(matches, 1):
                    label = star.proper or star.bf or star.gl or f"HIP{star.hip}"
                    print(f"{i}. {label} (HIP {star.hip or 'n/a'})")
                pick = int(input("Pick system number: ").strip())
                system_star = matches[pick - 1]
            else:
                system_star = matches[0]
            system_name = system_star.proper or system_star.bf or system_star.gl or f"HIP{system_star.hip}"
            system_xyz = (system_star.x, system_star.y, system_star.z)

    width = args.size
    height = args.height or args.size // 2
    label_entries = _load_label_entries(args, catalog, exclude_name=system_name)
    const_data = _load_constellation_overlays(args, catalog)
    img = render_skybox(
        catalog.arrays,
        system_xyz,
        width,
        height,
        args.mag_limit,
        args.background,
        args.background_projection,
        args.background_frame,
        args.star_frame,
        args.background_fill,
        (args.roll, args.pitch, args.yaw),
        args.splat_sigma,
        args.splat_min_sigma,
        not args.use_catalog_mag,
        args.background_destar,
        args.destar_percentile,
        args.destar_blur,
        args.destar_expand,
        args.background_gain,
        args.star_gain,
        args.exclude_nearby_pc,
        args.min_distance_ly / 3.26156 if args.min_distance_ly else 0.0,
        args.max_distance_ly / 3.26156 if args.max_distance_ly else 0.0,
        label_entries,
        args.label_color,
        args.label_font_size,
        not args.label_no_hip,
        args.label_alpha,
        args.overlay_guides,
        args.guide_color,
        args.guide_alpha,
        args.guide_meridian_step,
        args.guide_label_font_size,
        args.guide_label_lat,
        const_data[0],
        const_data[1],
        const_data[2],
        args.constellation_color,
        args.constellation_alpha,
        args.constellation_line_width,
        args.constellation_label_font_size,
        args.constellation_star_font_size,
        args.constellation_star_alpha,
        args.constellation_frame,
    )
    name = _sanitize_name(system_name)
    out_path = os.path.join(args.out, f"{name}_{width}x{height}.{args.format}")
    _save_image(img, out_path, args.format)
    print(f"Saved {out_path}")
    if args.convert_dds and args.format.lower() != "dds":
        dds_path = _convert_to_dds(
            out_path, args.convert_dds_out, args.dds_format, args.texconv_path
        )
        print(f"Saved {dds_path}")
    if system_star and args.caption:
        cap_path = os.path.join(args.out, f"{name}_caption.txt")
        apparent = None
        if args.constellations:
            table = load_constellation_table(args.constellations)
            apparent = apparent_constellation_for_star(table, system_xyz, system_star)
        write_caption(system_star, cap_path, system_name, apparent_constellation=apparent)
        print(f"Wrote {cap_path}")


def run_batch(args: argparse.Namespace) -> None:
    catalog, offset = _load_catalog_with_calibration(args)
    width = args.size
    height = args.height or args.size // 2
    if args.game_stars:
        game_entries = load_game_stars(args.game_stars)
        for entry in game_entries:
            system_name, system_xyz, star = resolve_game_star(catalog, entry)
            label_entries = _load_label_entries(args, catalog, exclude_name=system_name)
            const_data = _load_constellation_overlays(args, catalog)
            img = render_skybox(
                catalog.arrays,
                system_xyz,
                width,
                height,
                args.mag_limit,
                args.background,
                args.background_projection,
                args.background_frame,
                args.star_frame,
                args.background_fill,
                (args.roll, args.pitch, args.yaw),
                args.splat_sigma,
                args.splat_min_sigma,
                not args.use_catalog_mag,
                args.background_destar,
                args.destar_percentile,
                args.destar_blur,
                args.destar_expand,
                args.background_gain,
                args.star_gain,
                args.exclude_nearby_pc,
                args.min_distance_ly / 3.26156 if args.min_distance_ly else 0.0,
                args.max_distance_ly / 3.26156 if args.max_distance_ly else 0.0,
                label_entries,
                args.label_color,
                args.label_font_size,
                not args.label_no_hip,
                args.label_alpha,
                args.overlay_guides,
                args.guide_color,
                args.guide_alpha,
                args.guide_meridian_step,
                args.guide_label_font_size,
                args.guide_label_lat,
                const_data[0],
                const_data[1],
                const_data[2],
                args.constellation_color,
                args.constellation_alpha,
                args.constellation_line_width,
                args.constellation_label_font_size,
                args.constellation_star_font_size,
                args.constellation_star_alpha,
                args.constellation_frame,
            )
            name = _sanitize_name(system_name)
            out_path = os.path.join(args.out, f"{name}_{width}x{height}.{args.format}")
            _save_image(img, out_path, args.format)
            if args.convert_dds and args.format.lower() != "dds":
                _convert_to_dds(out_path, args.convert_dds_out, args.dds_format, args.texconv_path)
            if args.caption and star:
                cap_path = os.path.join(args.out, f"{name}_caption.txt")
                write_caption(star, cap_path, system_name)
        if args.constellations:
            out_path = os.path.join(args.out, "game_stars_captions.csv")
            names = [entry.hyg_query or entry.game_name for entry in game_entries if not entry.fictional]
            write_captions_for_names(
                catalog,
                names,
                out_path,
                conbound_c_path=args.constellations,
            )
        return

    if args.systems == "all":
        systems = catalog.stars
    else:
        systems = [s for s in catalog.stars if s.proper or s.bf or s.gl]
    if args.systems_list:
        names = read_name_list(args.systems_list)
        selected = []
        for name in names:
            matches = search_stars(catalog, name, limit=1)
            if matches:
                selected.append(matches[0])
        systems = selected

    for star in systems:
        system_name = star.proper or star.bf or star.gl or f"HIP{star.hip}"
        label_entries = _load_label_entries(args, catalog, exclude_name=system_name)
        const_data = _load_constellation_overlays(args, catalog)
        img = render_skybox(
            catalog.arrays,
            (star.x, star.y, star.z),
            width,
            height,
            args.mag_limit,
            args.background,
            args.background_projection,
            args.background_frame,
            args.star_frame,
            args.background_fill,
            (args.roll, args.pitch, args.yaw),
            args.splat_sigma,
            args.splat_min_sigma,
            not args.use_catalog_mag,
            args.background_destar,
            args.destar_percentile,
            args.destar_blur,
            args.destar_expand,
            args.background_gain,
            args.star_gain,
            args.exclude_nearby_pc,
            args.min_distance_ly / 3.26156 if args.min_distance_ly else 0.0,
            args.max_distance_ly / 3.26156 if args.max_distance_ly else 0.0,
            label_entries,
            args.label_color,
            args.label_font_size,
            not args.label_no_hip,
            args.label_alpha,
            args.overlay_guides,
            args.guide_color,
            args.guide_alpha,
            args.guide_meridian_step,
            args.guide_label_font_size,
            args.guide_label_lat,
            const_data[0],
            const_data[1],
            const_data[2],
            args.constellation_color,
            args.constellation_alpha,
            args.constellation_line_width,
            args.constellation_label_font_size,
            args.constellation_star_font_size,
            args.constellation_star_alpha,
            args.constellation_frame,
        )
        name = _sanitize_name(system_name)
        out_path = os.path.join(args.out, f"{name}_{width}x{height}.{args.format}")
        _save_image(img, out_path, args.format)
        if args.convert_dds and args.format.lower() != "dds":
            _convert_to_dds(out_path, args.convert_dds_out, args.dds_format, args.texconv_path)
        if args.caption:
            cap_path = os.path.join(args.out, f"{name}_caption.txt")
            write_caption(star, cap_path, system_name)

    if args.captions_list:
        out_path = os.path.join(args.out, "captions.csv")
        write_captions_for_names(
            catalog,
            read_name_list(args.captions_list),
            out_path,
            system_xyz=None,
            conbound_c_path=args.constellations,
        )


def run_captions(args: argparse.Namespace) -> None:
    catalog, offset = _load_catalog_with_calibration(args)
    out_path = args.out or "captions.csv"
    names = read_name_list(args.names)
    system_xyz = _parse_system_xyz(args, offset=offset)
    missing = write_captions_for_names(
        catalog,
        names,
        out_path,
        system_xyz=system_xyz,
        conbound_c_path=args.constellations,
    )
    print(f"Wrote {out_path}")
    if missing:
        print(f"Missing: {', '.join(missing)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Starfield skybox generator")
    parser.add_argument("--catalog", default="hygdata_v42.csv")

    subparsers = parser.add_subparsers(dest="command", required=True)

    interactive = subparsers.add_parser("interactive", help="Interactive skybox generator")
    _add_render_args(interactive)
    interactive.set_defaults(func=run_interactive)

    batch = subparsers.add_parser("batch", help="Batch skybox generator")
    _add_render_args(batch)
    batch.add_argument("--systems", choices=["named", "all"], default="named")
    batch.add_argument("--systems-list", help="File with system names to render")
    batch.add_argument("--captions-list", help="File with names for captions CSV")
    batch.set_defaults(func=run_batch)

    captions = subparsers.add_parser("captions", help="Generate caption CSV")
    captions.add_argument("--catalog", default="hygdata_v42.csv")
    captions.add_argument("--names", required=True, help="File with star names")
    captions.add_argument("--out", default="captions.csv")
    captions.add_argument("--constellations", default="constbnd/conbound.c")
    captions.add_argument("--system-x", type=float, default=None)
    captions.add_argument("--system-y", type=float, default=None)
    captions.add_argument("--system-z", type=float, default=None)
    captions.add_argument("--system-ra", type=float, default=None)
    captions.add_argument("--system-dec", type=float, default=None)
    captions.add_argument("--system-dist", type=float, default=None)
    captions.add_argument("--system-frame", choices=["equatorial", "galactic"], default="equatorial")
    _add_calibration_args(captions)
    captions.set_defaults(func=run_captions)

    import_sf = subparsers.add_parser("import-starfield", help="Convert Starfield system list to game stars CSV")
    import_sf.add_argument("--input", default="starfield_systems.txt")
    import_sf.add_argument("--out", default="data/game_stars.csv")
    import_sf.add_argument("--catalog", default="hygdata_v42.csv")
    import_sf.add_argument("--suggest-nearby", action="store_true")
    import_sf.add_argument("--suggest-distance-ly", type=float, default=1.0)
    import_sf.add_argument("--no-mark-fictional", action="store_true")
    _add_calibration_args(import_sf)
    import_sf.set_defaults(func=run_import_starfield)

    match = subparsers.add_parser("match-game-stars", help="Suggest HYG matches for game stars")
    match.add_argument("--game-stars", default="data/game_stars.csv")
    match.add_argument("--catalog", default="hygdata_v42.csv")
    match.add_argument("--out", default="data/game_stars_matches.csv")
    match.add_argument("--limit", type=int, default=5)
    match.set_defaults(func=run_match_game_stars)

    match_sf = subparsers.add_parser(
        "match-starfield", help="Suggest HYG matches using Starfield x/y/z coordinates"
    )
    match_sf.add_argument("--input", default="starfield_systems.txt")
    match_sf.add_argument("--catalog", default="hygdata_v42.csv")
    match_sf.add_argument("--out", default="data/starfield_matches.csv")
    match_sf.add_argument("--max-distance-ly", type=float, default=5.0)
    match_sf.add_argument("--name-weight", type=float, default=0.4)
    match_sf.add_argument("--apply", action="store_true")
    match_sf.add_argument("--game-stars", default="data/game_stars.csv")
    match_sf.add_argument("--apply-out", default="data/game_stars_with_matches.csv")
    _add_calibration_args(match_sf)
    match_sf.set_defaults(func=run_match_starfield)

    prep_bg = subparsers.add_parser(
        "preprocess-background", help="Convert all-sky background to equirectangular"
    )
    prep_bg.add_argument("--input", required=True)
    prep_bg.add_argument("--output", required=True)
    prep_bg.add_argument("--size", type=int, default=4096)
    prep_bg.add_argument("--height", type=int, default=None)
    prep_bg.add_argument(
        "--projection",
        default="mollweide",
        choices=["equirectangular", "mollweide"],
    )
    prep_bg.add_argument("--fill", choices=["wrap", "black"], default="wrap")
    prep_bg.add_argument("--gain", type=float, default=1.0)
    prep_bg.add_argument("--destar", action="store_true")
    prep_bg.add_argument("--destar-percentile", type=float, default=99.7)
    prep_bg.add_argument("--destar-blur", type=float, default=2.0)
    prep_bg.add_argument("--destar-expand", type=int, default=1)
    prep_bg.add_argument("--border-mask", action="store_true")
    prep_bg.add_argument("--border-tolerance", type=float, default=0.05)
    prep_bg.set_defaults(func=run_preprocess_background)

    return parser


def _add_render_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", default="out")
    parser.add_argument("--format", default="png", choices=["png", "jpg", "dds"])
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--mag-limit", type=float, default=7.5)
    parser.add_argument("--background", default=None)
    parser.add_argument(
        "--background-projection",
        default="mollweide",
        choices=["equirectangular", "mollweide"],
    )
    parser.add_argument(
        "--background-fill",
        default="wrap",
        choices=["wrap", "black"],
    )
    parser.add_argument("--background-gain", type=float, default=1.0)
    parser.add_argument("--star-gain", type=float, default=0.2)
    parser.add_argument("--exclude-nearby-pc", type=float, default=0.05)
    parser.add_argument("--min-distance-ly", type=float, default=0.0)
    parser.add_argument("--max-distance-ly", type=float, default=0.0)
    parser.add_argument(
        "--background-frame",
        default="galactic",
        choices=["equatorial", "galactic"],
    )
    parser.add_argument(
        "--star-frame",
        default="equatorial",
        choices=["equatorial", "galactic"],
    )
    parser.add_argument("--background-destar", action="store_true")
    parser.add_argument("--destar-percentile", type=float, default=99.7)
    parser.add_argument("--destar-blur", type=float, default=2.0)
    parser.add_argument("--destar-expand", type=int, default=1)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--splat-sigma", type=float, default=2.0)
    parser.add_argument("--splat-min-sigma", type=float, default=0.6)
    parser.add_argument(
        "--use-catalog-mag",
        action="store_true",
        help="Use catalog apparent magnitude instead of computed apparent magnitude",
    )
    parser.add_argument("--caption", action="store_true")
    parser.add_argument("--system-name", default=None)
    parser.add_argument("--system-hip", type=int, default=None)
    parser.add_argument("--system-x", type=float, default=None)
    parser.add_argument("--system-y", type=float, default=None)
    parser.add_argument("--system-z", type=float, default=None)
    parser.add_argument("--system-ra", type=float, default=None)
    parser.add_argument("--system-dec", type=float, default=None)
    parser.add_argument("--system-dist", type=float, default=None)
    parser.add_argument("--system-frame", choices=["equatorial", "galactic"], default="equatorial")
    parser.add_argument("--constellations", default="constbnd/conbound.c")
    parser.add_argument("--game-stars", help="CSV list of game stars to render")
    parser.add_argument("--label-game-stars", help="CSV list of game stars to label")
    parser.add_argument("--label-color", default="#70ff70")
    parser.add_argument("--label-font-size", type=int, default=18)
    parser.add_argument("--label-alpha", type=int, default=200)
    parser.add_argument("--label-no-hip", action="store_true")
    parser.add_argument("--overlay-guides", action="store_true")
    parser.add_argument("--guide-color", default="#70ff70")
    parser.add_argument("--guide-alpha", type=int, default=140)
    parser.add_argument("--guide-meridian-step", type=float, default=15.0)
    parser.add_argument("--guide-label-font-size", type=int, default=14)
    parser.add_argument("--guide-label-lat", type=float, default=60.0)
    parser.add_argument("--overlay-constellations", action="store_true")
    parser.add_argument("--constellation-lines", default="data/ConstellationLines.csv")
    parser.add_argument("--constellation-color", default="#70ff70")
    parser.add_argument("--constellation-alpha", type=int, default=220)
    parser.add_argument("--constellation-line-width", type=int, default=2)
    parser.add_argument("--constellation-label-font-size", type=int, default=14)
    parser.add_argument("--constellation-star-labels", action="store_true")
    parser.add_argument("--constellation-star-font-size", type=int, default=12)
    parser.add_argument("--constellation-star-alpha", type=int, default=160)
    parser.add_argument("--constellation-star-mag-limit", type=float, default=2.5)
    parser.add_argument("--constellation-frame", choices=["equatorial", "galactic"], default="equatorial")
    parser.add_argument("--convert-dds", action="store_true")
    parser.add_argument("--convert-dds-out", default="out_dds")
    parser.add_argument("--dds-format", default="BC7_UNORM")
    parser.add_argument("--texconv-path", default=None)
    _add_calibration_args(parser)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


def run_import_starfield(args: argparse.Namespace) -> None:
    systems = load_starfield_systems(args.input)
    if not systems:
        raise SystemExit("No systems found in input file.")
    write_game_stars_csv(systems, args.out, mark_fictional_with_xyz=not args.no_mark_fictional)
    print(f"Wrote {args.out}")
    _write_coordinate_deltas(systems, args, args.out.replace(".csv", "_coord_deltas.csv"))
    if args.suggest_nearby:
        catalog, _ = _load_catalog_with_calibration(args)
        _write_nearby_suggestions(
            systems,
            catalog,
            args.out.replace(".csv", "_suggestions.csv"),
            args.suggest_distance_ly,
        )


def run_match_game_stars(args: argparse.Namespace) -> None:
    catalog = load_hyg_catalog(args.catalog)
    name_index = build_name_index(catalog)
    stars = load_game_stars(args.game_stars)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("game_name,suggest_name,hip,proper,bf,gl\n")
        for entry in stars:
            if entry.fictional:
                continue
            if entry.hip or entry.hyg_query:
                continue
            query = entry.game_name.strip().lower()
            if not query:
                continue
            candidates = difflib.get_close_matches(query, list(name_index.keys()), n=args.limit, cutoff=0.6)
            for cand in candidates:
                for star in name_index.get(cand, [])[:1]:
                    name = star.proper or star.bf or star.gl or cand
                    f.write(f"{entry.game_name},{name},{star.hip or ''},{star.proper},{star.bf},{star.gl}\n")
    print(f"Wrote {args.out}")


def run_match_starfield(args: argparse.Namespace) -> None:
    catalog, _ = _load_catalog_with_calibration(args)
    systems = load_starfield_systems(args.input)
    name_index = build_name_index(catalog)
    max_distance_pc = args.max_distance_ly / 3.26156
    matches = {}
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("game_name,suggest_name,hip,dist_ly,name_score,combined_score\n")
        for system in systems:
            if system.x is None or system.y is None or system.z is None:
                continue
            query = system.name.strip().lower()
            dx = catalog.arrays["x"] - system.x
            dy = catalog.arrays["y"] - system.y
            dz = catalog.arrays["z"] - system.z
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            idx = int(np.argmin(dist))
            if dist[idx] > max_distance_pc:
                continue
            star = catalog.stars[idx]
            name = star.proper or star.bf or star.gl or f"HIP{star.hip}"
            name_score = 0.0
            if query:
                candidates = difflib.get_close_matches(query, list(name_index.keys()), n=1, cutoff=0.0)
                if candidates:
                    name_score = difflib.SequenceMatcher(None, query, candidates[0]).ratio()
            dist_score = max(0.0, 1.0 - (dist[idx] / max_distance_pc))
            combined = (1 - args.name_weight) * dist_score + args.name_weight * name_score
            f.write(
                f"{system.name},{name},{star.hip or ''},{dist[idx] * 3.26156:.3f},{name_score:.3f},{combined:.3f}\n"
            )
            matches[system.name] = (name, star.hip)
    print(f"Wrote {args.out}")
    if args.apply:
        _apply_matches_to_game_stars(args.game_stars, args.apply_out, matches)
        print(f"Wrote {args.apply_out}")


def _apply_matches_to_game_stars(path: str, out_path: str, matches: dict) -> None:
    # Attach matched HYG identifiers to the game stars table without overwriting explicit values.
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        fieldnames = list(reader.fieldnames or [])
        for extra in ["hyg_match", "hip", "hyg_query"]:
            if extra not in fieldnames:
                fieldnames.append(extra)
        rows = []
        for row in reader:
            name = (row.get("game_name") or "").strip()
            if name in matches:
                match_name, match_hip = matches[name]
                row["hyg_match"] = f"{match_name}|{match_hip or ''}"
                if not (row.get("hip") or "").strip() and match_hip:
                    row["hip"] = str(match_hip)
                if not (row.get("hyg_query") or "").strip():
                    row["hyg_query"] = match_name
            rows.append(row)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_preprocess_background(args: argparse.Namespace) -> None:
    height = args.height or args.size // 2
    preprocess_background(
        args.input,
        args.output,
        args.size,
        height,
        args.projection,
        args.fill,
        args.gain,
        args.destar,
        args.destar_percentile,
        args.destar_blur,
        args.destar_expand,
        args.border_mask,
        args.border_tolerance,
    )
    print(f"Wrote {args.output}")


def _add_calibration_args(parser: argparse.ArgumentParser) -> None:
    # Default calibration uses Alpha Centauri to align HYG with in-game coordinates.
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--calibration-name", default="Alpha Centauri")
    parser.add_argument("--calibration-x", type=float, default=-0.4952)
    parser.add_argument("--calibration-y", type=float, default=-0.4141)
    parser.add_argument("--calibration-z", type=float, default=-1.1566)


def _load_catalog_with_calibration(args: argparse.Namespace):
    catalog = load_hyg_catalog(args.catalog)
    if not args.no_calibration:
        offset = compute_offset(
            catalog,
            args.calibration_name,
            (args.calibration_x, args.calibration_y, args.calibration_z),
        )
        apply_offset(catalog, offset)
        return catalog, offset
    return catalog, (0.0, 0.0, 0.0)


def _load_label_entries(
    args: argparse.Namespace, catalog, exclude_name: Optional[str] = None
) -> Optional[list]:
    if not args.label_game_stars:
        return None
    entries = []
    game_entries = load_game_stars(args.label_game_stars)
    for entry in game_entries:
        try:
            name, xyz, star = resolve_game_star(catalog, entry)
            if exclude_name and name.lower() == exclude_name.lower():
                continue
            hip = star.hip if star else None
            entries.append((name, hip, xyz))
        except ValueError:
            continue
    return entries


def _load_constellation_overlays(args: argparse.Namespace, catalog):
    if not args.overlay_constellations:
        return (None, None, None)
    segments, const_labels, star_labels = build_constellation_overlays(
        catalog,
        args.constellation_lines,
        args.constellation_star_mag_limit,
    )
    if not args.constellation_star_labels:
        star_labels = []
    return (segments, const_labels, star_labels)


def _write_nearby_suggestions(systems, catalog, out_path: str, max_distance_ly: float) -> None:
    # Suggest real HYG stars within a small radius for fictional systems.
    max_distance_pc = max_distance_ly / 3.26156
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("game_name,nearest_name,nearest_hip,distance_ly\n")
        for system in systems:
            if system.x is None or system.y is None or system.z is None:
                continue
            dx = catalog.arrays["x"] - system.x
            dy = catalog.arrays["y"] - system.y
            dz = catalog.arrays["z"] - system.z
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)
            idx = int(np.argmin(dist))
            if dist[idx] <= max_distance_pc:
                star = catalog.stars[idx]
                name = star.proper or star.bf or star.gl or f"HIP{star.hip}"
                f.write(f"{system.name},{name},{star.hip or ''},{dist[idx] * 3.26156:.3f}\n")
    print(f"Wrote {out_path}")


def _write_coordinate_deltas(systems, args: argparse.Namespace, out_path: str) -> None:
    # Show how far the game coordinates are from the matched HYG star after calibration.
    catalog, _ = _load_catalog_with_calibration(args)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("game_name,match_name,match_hip,dx,dy,dz,dist_pc,dist_ly\n")
        for system in systems:
            if system.x is None or system.y is None or system.z is None:
                continue
            matches = search_stars(catalog, system.name, limit=1)
            if not matches:
                f.write(f"{system.name},,,,,,,\n")
                continue
            star = matches[0]
            dx = system.x - star.x
            dy = system.y - star.y
            dz = system.z - star.z
            dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            name = star.proper or star.bf or star.gl or f"HIP{star.hip}"
            f.write(
                f"{system.name},{name},{star.hip or ''},{dx:.4f},{dy:.4f},{dz:.4f},{dist:.4f},{dist * 3.26156:.4f}\n"
            )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
