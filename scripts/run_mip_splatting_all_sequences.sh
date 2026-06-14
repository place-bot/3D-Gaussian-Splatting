#!/usr/bin/env bash
set -euo pipefail

# Batch runner for applying the tuned Mip-Splatting setup to all Stage 2
# sequences. This script is intended for Linux servers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-light}"
SWEEP_SCRIPT="${REPO_ROOT}/scripts/run_mip_splatting_sweep.sh"
RANK_BY="${RANK_BY:-mean_psnr}"
RESOLUTION="${RESOLUTION:-1}"
SKIP_MISSING="${SKIP_MISSING:-1}"
BASE_PORT="${BASE_PORT:-6109}"

# Space-separated subset override, for example:
#   SEQUENCES="Sequence_01 Sequence_03" bash scripts/run_mip_splatting_all_sequences.sh light
SEQUENCES="${SEQUENCES:-Sequence_01 Sequence_02 Sequence_03 Sequence_04 Sequence_05}"

# Current best neighborhood. These defaults keep the tuning local around the
# best Sequence_04 result instead of running a broad expensive search.
LIGHT_KERNEL_SIZES="${LIGHT_KERNEL_SIZES:-0.15 0.2 0.25}"
LIGHT_LAMBDA_DSSIMS="${LIGHT_LAMBDA_DSSIMS:-0.05 0.1}"
LIGHT_DENSIFY_UNTILS="${LIGHT_DENSIFY_UNTILS:-12000 15000 17000}"
LIGHT_DENSIFY_GRADS="${LIGHT_DENSIFY_GRADS:-0.0002 0.00025}"

BEST_KERNEL_SIZE="${BEST_KERNEL_SIZE:-0.2}"
BEST_LAMBDA_DSSIM="${BEST_LAMBDA_DSSIM:-0.1}"
BEST_DENSIFY_UNTIL="${BEST_DENSIFY_UNTIL:-15000}"
BEST_DENSIFY_GRAD="${BEST_DENSIFY_GRAD:-0.0002}"
ITERATIONS="${ITERATIONS:-30000}"

usage() {
  cat <<USAGE
Usage:
  bash scripts/run_mip_splatting_all_sequences.sh [best|light|coarse|densify]

Modes:
  best     Run one current-best Mip-Splatting configuration per sequence.
  light    Run local coarse sweep, then local densify sweep per sequence.
  coarse   Run only local kernel/lambda sweep per sequence.
  densify  Run only local densify sweep per sequence; requires prior coarse.

Default source variants:
  Sequence_01 -> enhanced_long
  Sequence_02 -> partition_mid
  Sequence_03 -> partition_best
  Sequence_04 -> partition_best
  Sequence_05 -> full

Useful environment variables:
  CUDA_VISIBLE_DEVICES=0
  RANK_BY=mean_psnr
  SEQUENCES="Sequence_01 Sequence_02"
  SKIP_MISSING=1
  ITERATIONS=30000
  RESOLUTION=1
  OVERWRITE=1
  LIGHT_KERNEL_SIZES="0.15 0.2 0.25"
  LIGHT_LAMBDA_DSSIMS="0.05 0.1"
  LIGHT_DENSIFY_UNTILS="12000 15000 17000"
  LIGHT_DENSIFY_GRADS="0.0002 0.00025"
USAGE
}

variant_for_sequence() {
  case "$1" in
    Sequence_01) echo "enhanced_long" ;;
    Sequence_02) echo "partition_mid" ;;
    Sequence_03) echo "partition_best" ;;
    Sequence_04) echo "partition_best" ;;
    Sequence_05) echo "full" ;;
    *)
      echo "Unknown sequence: $1" >&2
      return 1
      ;;
  esac
}

check_source_or_skip() {
  local sequence="$1"
  local variant="$2"
  local undistorted_dir="${REPO_ROOT}/work/${sequence}/${variant}/undistorted"
  if [[ -d "${undistorted_dir}/images" && -d "${undistorted_dir}/sparse" ]]; then
    return 0
  fi
  local message="Missing COLMAP undistorted source for ${sequence}: ${undistorted_dir}"
  if [[ "${SKIP_MISSING}" == "1" ]]; then
    echo "WARNING: ${message}; skipping ${sequence}."
    return 1
  fi
  echo "ERROR: ${message}" >&2
  exit 1
}

run_sequence_best() {
  local sequence="$1"
  local variant="$2"
  local out_root="${REPO_ROOT}/work/${sequence}/mip_splatting_best"
  echo
  echo "=== ${sequence}: current best config -> ${out_root} ==="
  SEQUENCE="${sequence}" \
  SOURCE_VARIANT="${variant}" \
  OUT_ROOT="${out_root}" \
  RESOLUTION="${RESOLUTION}" \
  RANK_BY="${RANK_BY}" \
  BASE_PORT="${BASE_PORT}" \
  KERNEL_SIZES="${BEST_KERNEL_SIZE}" \
  LAMBDA_DSSIMS="${BEST_LAMBDA_DSSIM}" \
  COARSE_ITERATIONS="${ITERATIONS}" \
  COARSE_DENSIFY_UNTIL="${BEST_DENSIFY_UNTIL}" \
  COARSE_DENSIFY_GRAD="${BEST_DENSIFY_GRAD}" \
  OVERWRITE="${OVERWRITE:-0}" \
  bash "${SWEEP_SCRIPT}" coarse
}

run_sequence_coarse() {
  local sequence="$1"
  local variant="$2"
  local out_root="${REPO_ROOT}/work/${sequence}/mip_splatting_light"
  echo
  echo "=== ${sequence}: local kernel/lambda sweep -> ${out_root} ==="
  SEQUENCE="${sequence}" \
  SOURCE_VARIANT="${variant}" \
  OUT_ROOT="${out_root}" \
  RESOLUTION="${RESOLUTION}" \
  RANK_BY="${RANK_BY}" \
  BASE_PORT="${BASE_PORT}" \
  KERNEL_SIZES="${LIGHT_KERNEL_SIZES}" \
  LAMBDA_DSSIMS="${LIGHT_LAMBDA_DSSIMS}" \
  COARSE_ITERATIONS="${ITERATIONS}" \
  COARSE_DENSIFY_UNTIL="${BEST_DENSIFY_UNTIL}" \
  COARSE_DENSIFY_GRAD="${BEST_DENSIFY_GRAD}" \
  OVERWRITE="${OVERWRITE:-0}" \
  bash "${SWEEP_SCRIPT}" coarse
}

run_sequence_densify() {
  local sequence="$1"
  local variant="$2"
  local out_root="${REPO_ROOT}/work/${sequence}/mip_splatting_light"
  echo
  echo "=== ${sequence}: local densify sweep -> ${out_root} ==="
  SEQUENCE="${sequence}" \
  SOURCE_VARIANT="${variant}" \
  OUT_ROOT="${out_root}" \
  RESOLUTION="${RESOLUTION}" \
  RANK_BY="${RANK_BY}" \
  BASE_PORT="${BASE_PORT}" \
  DENSIFY_ITERATIONS="${ITERATIONS}" \
  DENSIFY_UNTILS="${LIGHT_DENSIFY_UNTILS}" \
  DENSIFY_GRADS="${LIGHT_DENSIFY_GRADS}" \
  OVERWRITE="${OVERWRITE:-0}" \
  bash "${SWEEP_SCRIPT}" densify
}

case "${MODE}" in
  -h|--help|help)
    usage
    exit 0
    ;;
  best|light|coarse|densify)
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac

if [[ ! -f "${SWEEP_SCRIPT}" ]]; then
  echo "Missing sweep script: ${SWEEP_SCRIPT}" >&2
  exit 1
fi

for sequence in ${SEQUENCES}; do
  variant="$(variant_for_sequence "${sequence}")"
  if ! check_source_or_skip "${sequence}" "${variant}"; then
    continue
  fi

  case "${MODE}" in
    best)
      run_sequence_best "${sequence}" "${variant}"
      ;;
    coarse)
      run_sequence_coarse "${sequence}" "${variant}"
      ;;
    densify)
      run_sequence_densify "${sequence}" "${variant}"
      ;;
    light)
      run_sequence_coarse "${sequence}" "${variant}"
      run_sequence_densify "${sequence}" "${variant}"
      ;;
  esac
done

echo
echo "All requested sequences finished."
