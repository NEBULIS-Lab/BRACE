#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="Lift"
ROBOTS=("Panda")
CAMERA="agentview"
STEPS="200"
FPS="20"
WIDTH="256"
HEIGHT="256"
OUT="artifacts/robosuite/rollout.mp4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env) ENV_NAME="$2"; shift 2 ;;
    --robots)
      shift
      ROBOTS=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do ROBOTS+=("$1"); shift; done
      ;;
    --camera) CAMERA="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --height) HEIGHT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: scripts/run_robosuite_render.sh [--env <name>] [--robots <r1> <r2> ...] [--camera <cam>] [--steps <n>] [--out <file>]

Requires:
  - a working RoboSuite installation in the active Python env
  - MuJoCo headless rendering support (EGL recommended)
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

python experiments/robosuite/render_offscreen_video.py \
  --env "$ENV_NAME" \
  --robots "${ROBOTS[@]}" \
  --camera "$CAMERA" \
  --steps "$STEPS" \
  --fps "$FPS" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --out "$OUT"
