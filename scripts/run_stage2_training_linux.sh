#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEQUENCE="${SEQUENCE:-Sequence_01}"
VARIANT="${VARIANT:-improved}"
SOURCE_VARIANT="${SOURCE_VARIANT:-$VARIANT}"
OUTPUT_VARIANT="${OUTPUT_VARIANT:-$VARIANT}"
ITERATIONS="${ITERATIONS:-9000}"
MAX_SIDE="${MAX_SIDE:-960}"
DENSIFY_UNTIL="${DENSIFY_UNTIL:-3500}"
DENSIFY_EVERY="${DENSIFY_EVERY:-500}"
DENSIFY_FRACTION="${DENSIFY_FRACTION:-0.08}"
DENSIFY_MAX_NEW="${DENSIFY_MAX_NEW:-1600}"
LAMBDA_SSIM="${LAMBDA_SSIM:-0.0}"
LAMBDA_EDGE="${LAMBDA_EDGE:-0.0}"
LAMBDA_OPACITY="${LAMBDA_OPACITY:-0.0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
ALLOW_RESUME_RESOLUTION_MISMATCH="${ALLOW_RESUME_RESOLUTION_MISMATCH:-0}"
LR_SCALE="${LR_SCALE:-1.0}"
SAVE_RENDER_COUNT="${SAVE_RENDER_COUNT:-16}"
PYTHON_BIN="${PYTHON_BIN:-python}"

SCENE_DIR="$ROOT/work/$SEQUENCE/$SOURCE_VARIANT/undistorted"
OUTPUT_DIR="$ROOT/work/$SEQUENCE/$OUTPUT_VARIANT/output"
FIGURES_DIR="$ROOT/work/$SEQUENCE/$OUTPUT_VARIANT/figures"
TRAIN_SCRIPT="$ROOT/scripts/train_stage2_gsplat.py"
LOG_PATH="$ROOT/work/$SEQUENCE/$OUTPUT_VARIANT/training.log"

if [[ ! -d "$SCENE_DIR" ]]; then
  echo "Scene directory does not exist: $SCENE_DIR" >&2
  exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Training script does not exist: $TRAIN_SCRIPT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR" "$FIGURES_DIR"

cmd=(
  "$PYTHON_BIN" "$TRAIN_SCRIPT"
  --scene-dir "$SCENE_DIR"
  --output-dir "$OUTPUT_DIR"
  --figures-dir "$FIGURES_DIR"
  --iterations "$ITERATIONS"
  --max-side "$MAX_SIDE"
  --save-render-count "$SAVE_RENDER_COUNT"
  --densify-until "$DENSIFY_UNTIL"
  --densify-every "$DENSIFY_EVERY"
  --densify-fraction "$DENSIFY_FRACTION"
  --densify-max-new "$DENSIFY_MAX_NEW"
  --lambda-ssim "$LAMBDA_SSIM"
  --lambda-edge "$LAMBDA_EDGE"
  --lambda-opacity "$LAMBDA_OPACITY"
  --lr-scale "$LR_SCALE"
)

if [[ -n "$RESUME_CHECKPOINT" ]]; then
  if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
    echo "Resume checkpoint does not exist: $RESUME_CHECKPOINT" >&2
    exit 1
  fi
  cmd+=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

if [[ "$ALLOW_RESUME_RESOLUTION_MISMATCH" == "1" ]]; then
  cmd+=(--allow-resume-resolution-mismatch)
fi

echo "Running: ${cmd[*]}"
"${cmd[@]}" > "$LOG_PATH" 2>&1
echo "Training log: $LOG_PATH"
