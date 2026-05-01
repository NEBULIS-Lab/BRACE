#!/usr/bin/env bash
set -euo pipefail

EP_DIR=""
ENV_NAME="Lift"
ROBOTS=("Panda")
CAMERA="agentview"
WIDTH="512"
HEIGHT="512"
FPS="20"
OUT="artifacts/robosuite/playback.mp4"
MAX_FRAMES="600"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ep-dir) EP_DIR="$2"; shift 2 ;;
    --env) ENV_NAME="$2"; shift 2 ;;
    --robots)
      shift
      ROBOTS=()
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do ROBOTS+=("$1"); shift; done
      ;;
    --camera) CAMERA="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --height) HEIGHT="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    -h|--help)
      cat <<EOF
Usage: scripts/run_robosuite_playback.sh --ep-dir <ep_dir> [--env <name>] [--robots <r1> <r2> ...] [--out <file>]
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$EP_DIR" ]]; then
  echo "--ep-dir is required" >&2
  exit 2
fi

python experiments/robosuite/playback_npz_demo_to_video.py \
  --ep-dir "$EP_DIR" \
  --env "$ENV_NAME" \
  --robots "${ROBOTS[@]}" \
  --camera "$CAMERA" \
  --width "$WIDTH" \
  --height "$HEIGHT" \
  --fps "$FPS" \
  --out "$OUT" \
  --max-frames "$MAX_FRAMES"
