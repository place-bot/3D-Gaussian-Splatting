#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Set MODE=from_scratch to train a new high-resolution model from COLMAP points.
# Default mode resumes the existing lower-resolution finetune checkpoint and
# continues optimization at a higher image resolution.
MODE="${MODE:-resume}"
MAX_SIDE="${MAX_SIDE:-1280}"
PYTHON_BIN="${PYTHON_BIN:-python}"

case "$MODE" in
  resume)
    SEQUENCE=Sequence_04 \
    SOURCE_VARIANT=partition_best \
    OUTPUT_VARIANT="partition_best_highres${MAX_SIDE}_resume" \
    ITERATIONS="${ITERATIONS:-6000}" \
    MAX_SIDE="$MAX_SIDE" \
    DENSIFY_UNTIL="${DENSIFY_UNTIL:-0}" \
    DENSIFY_EVERY="${DENSIFY_EVERY:-500}" \
    DENSIFY_FRACTION="${DENSIFY_FRACTION:-0.04}" \
    DENSIFY_MAX_NEW="${DENSIFY_MAX_NEW:-600}" \
    LAMBDA_SSIM="${LAMBDA_SSIM:-0.05}" \
    LAMBDA_EDGE="${LAMBDA_EDGE:-0.01}" \
    LAMBDA_OPACITY="${LAMBDA_OPACITY:-0.0005}" \
    LR_SCALE="${LR_SCALE:-0.20}" \
    SAVE_RENDER_COUNT="${SAVE_RENDER_COUNT:-24}" \
    RESUME_CHECKPOINT="$ROOT/work/Sequence_04/partition_best_finetune/output/model_final.pt" \
    ALLOW_RESUME_RESOLUTION_MISMATCH=1 \
    PYTHON_BIN="$PYTHON_BIN" \
    "$ROOT/scripts/run_stage2_training_linux.sh"
    ;;
  from_scratch)
    SEQUENCE=Sequence_04 \
    SOURCE_VARIANT=partition_best \
    OUTPUT_VARIANT="partition_best_highres${MAX_SIDE}_scratch" \
    ITERATIONS="${ITERATIONS:-18000}" \
    MAX_SIDE="$MAX_SIDE" \
    DENSIFY_UNTIL="${DENSIFY_UNTIL:-9000}" \
    DENSIFY_EVERY="${DENSIFY_EVERY:-400}" \
    DENSIFY_FRACTION="${DENSIFY_FRACTION:-0.08}" \
    DENSIFY_MAX_NEW="${DENSIFY_MAX_NEW:-1200}" \
    LAMBDA_SSIM="${LAMBDA_SSIM:-0.05}" \
    LAMBDA_EDGE="${LAMBDA_EDGE:-0.01}" \
    LAMBDA_OPACITY="${LAMBDA_OPACITY:-0.0005}" \
    LR_SCALE="${LR_SCALE:-1.0}" \
    SAVE_RENDER_COUNT="${SAVE_RENDER_COUNT:-24}" \
    PYTHON_BIN="$PYTHON_BIN" \
    "$ROOT/scripts/run_stage2_training_linux.sh"
    ;;
  *)
    echo "Unknown MODE=$MODE. Use MODE=resume or MODE=from_scratch." >&2
    exit 1
    ;;
esac
