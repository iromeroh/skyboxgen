#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat << 'USAGE'
Generate 8K skybox PNG+DDS for Starfield systems that have XYZ coordinates in data/game_stars.csv.

Usage:
  scripts/generate_starfield_xyz_skyboxes.sh [validation|production] [--force] [--list-only]

Modes:
  validation  Adds labels + meridians/guides + constellation overlays for visual verification.
  production  Clean output: no labels, no meridians/guides, no constellations.

Env overrides:
  PYTHON_BIN          (default: .venv/bin/python)
  GAME_STARS_CSV      (default: data/game_stars.csv)
  SIZE                (default: 8192)
  HEIGHT              (default: 4096)
  MAG_LIMIT           (default: 7.5)
  BACKGROUND_PATH     (default: Allsky-Stars_full.jpg)
  BACKGROUND_PROJ     (default: mollweide)
  BACKGROUND_FILL     (default: wrap)
  TEXCONV_PATH        (default: texconv/texconv.exe)
  OUT_DIR             (default: out_validation|out by mode)
  DDS_DIR             (default: out_dds_validation|out_dds by mode)
USAGE
}

MODE="${1:-validation}"
if [[ "${MODE}" == "-h" || "${MODE}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "${MODE}" != "validation" && "${MODE}" != "production" ]]; then
  echo "Invalid mode: ${MODE}" >&2
  usage
  exit 1
fi
shift || true

FORCE=0
LIST_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1 ;;
    --list-only) LIST_ONLY=1 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
GAME_STARS_CSV="${GAME_STARS_CSV:-${ROOT_DIR}/data/game_stars.csv}"
SIZE="${SIZE:-8192}"
HEIGHT="${HEIGHT:-4096}"
MAG_LIMIT="${MAG_LIMIT:-7.5}"
BACKGROUND_PATH="${BACKGROUND_PATH:-${ROOT_DIR}/Allsky-Stars_full.jpg}"
BACKGROUND_PROJ="${BACKGROUND_PROJ:-mollweide}"
BACKGROUND_FILL="${BACKGROUND_FILL:-wrap}"
TEXCONV_PATH="${TEXCONV_PATH:-${ROOT_DIR}/texconv/texconv.exe}"

if [[ "${MODE}" == "validation" ]]; then
  OUT_DIR_DEFAULT="${ROOT_DIR}/out_validation"
  DDS_DIR_DEFAULT="${ROOT_DIR}/out_dds_validation"
else
  OUT_DIR_DEFAULT="${ROOT_DIR}/out"
  DDS_DIR_DEFAULT="${ROOT_DIR}/out_dds"
fi
OUT_DIR="${OUT_DIR:-${OUT_DIR_DEFAULT}}"
DDS_DIR="${DDS_DIR:-${DDS_DIR_DEFAULT}}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -f "${GAME_STARS_CSV}" ]]; then
  echo "Missing game stars CSV: ${GAME_STARS_CSV}" >&2
  exit 1
fi
if [[ ! -f "${BACKGROUND_PATH}" ]]; then
  echo "Missing background image: ${BACKGROUND_PATH}" >&2
  exit 1
fi
if [[ ! -f "${TEXCONV_PATH}" ]]; then
  echo "Missing texconv path: ${TEXCONV_PATH}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}" "${DDS_DIR}"

mapfile -t SYSTEM_ROWS < <("${PYTHON_BIN}" - << 'PY' "${GAME_STARS_CSV}"
import csv
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
with csv_path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    rows = []
    for row in reader:
        name = (row.get("game_name") or "").strip()
        x = (row.get("x") or "").strip()
        y = (row.get("y") or "").strip()
        z = (row.get("z") or "").strip()
        if not name or not x or not y or not z:
            continue
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:64] or "system"
        rows.append((name, safe, x, y, z))

for name, safe, x, y, z in rows:
    print("\t".join([name, safe, x, y, z]))
PY
)

TOTAL="${#SYSTEM_ROWS[@]}"
echo "Mode: ${MODE}"
echo "Systems with XYZ: ${TOTAL}"
echo "PNG output: ${OUT_DIR}"
echo "DDS output: ${DDS_DIR}"

if [[ "${LIST_ONLY}" -eq 1 ]]; then
  printf '%s\n' "${SYSTEM_ROWS[@]}" | cut -f1
  exit 0
fi

COMMON_ARGS=(
  -m skyboxgen.cli interactive
  --catalog "${ROOT_DIR}/hygdata_v42.csv"
  --background "${BACKGROUND_PATH}"
  --background-projection "${BACKGROUND_PROJ}"
  --background-fill "${BACKGROUND_FILL}"
  --background-destar
  --mag-limit "${MAG_LIMIT}"
  --size "${SIZE}"
  --height "${HEIGHT}"
  --format png
  --out "${OUT_DIR}"
  --convert-dds
  --convert-dds-out "${DDS_DIR}"
  --dds-format BC7_UNORM
  --texconv-path "${TEXCONV_PATH}"
)

if [[ "${MODE}" == "validation" ]]; then
  MODE_ARGS=(
    --label-game-stars "${GAME_STARS_CSV}"
    --label-color "#ffd64d"
    --label-no-hip
    --overlay-guides
    --guide-color "#5bff8a"
    --guide-meridian-step 15
    --overlay-constellations
    --constellation-frame equatorial
    --constellation-color "#31b8ff"
    --constellation-alpha 220
    --constellation-line-width 2
    --constellation-star-labels
  )
else
  MODE_ARGS=()
fi

DONE=0
SKIP=0
FAIL=0

for row in "${SYSTEM_ROWS[@]}"; do
  IFS=$'\t' read -r NAME SAFE X Y Z <<< "${row}"
  PNG_PATH="${OUT_DIR}/${SAFE}_${SIZE}x${HEIGHT}.png"
  DDS_PATH="${DDS_DIR}/${SAFE}_${SIZE}x${HEIGHT}.dds"

  if [[ "${FORCE}" -eq 0 && -f "${PNG_PATH}" && -f "${DDS_PATH}" ]]; then
    echo "[SKIP] ${NAME}"
    SKIP=$((SKIP + 1))
    continue
  fi

  echo "[RUN] ${NAME}"
  if "${PYTHON_BIN}" "${COMMON_ARGS[@]}" "${MODE_ARGS[@]}" \
      --system-name "${NAME}" --system-x "${X}" --system-y "${Y}" --system-z "${Z}" --system-frame equatorial; then
    DONE=$((DONE + 1))
  else
    echo "[FAIL] ${NAME}" >&2
    FAIL=$((FAIL + 1))
  fi
done

echo "Completed: done=${DONE} skip=${SKIP} fail=${FAIL} total=${TOTAL}"
[[ "${FAIL}" -eq 0 ]]
