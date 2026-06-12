#!/usr/bin/env bash
set -euo pipefail

# Mip-Splatting parameter sweep for the Stage 2 COLMAP datasets.
# Run from the repository root, or call this script by absolute path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

MODE="${1:-coarse}"

SEQUENCE="${SEQUENCE:-Sequence_04}"
SOURCE_VARIANT="${SOURCE_VARIANT:-partition_best}"
MIP_DIR="${MIP_DIR:-${REPO_ROOT}/external/mip-splatting-main}"
SOURCE_DIR="${SOURCE_DIR:-${REPO_ROOT}/work/${SEQUENCE}/${SOURCE_VARIANT}_mip_input}"
UNDISTORTED_DIR="${UNDISTORTED_DIR:-${REPO_ROOT}/work/${SEQUENCE}/${SOURCE_VARIANT}/undistorted}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/work/${SEQUENCE}/mip_splatting_sweep}"
RESOLUTION="${RESOLUTION:-1}"
BASE_PORT="${BASE_PORT:-6109}"
RANK_BY="${RANK_BY:-mean_ssim}"
OVERWRITE="${OVERWRITE:-0}"

RUN_INDEX=0
SUMMARY_CSV="${OUT_ROOT}/sweep_summary.csv"

usage() {
  cat <<USAGE
Usage:
  bash scripts/run_mip_splatting_sweep.sh [coarse|densify|long|all]

Modes:
  coarse   Run 9 jobs: kernel_size x lambda_dssim at fixed densification settings.
  densify  Use best coarse kernel/lambda, then sweep densify_until_iter and densify_grad_threshold.
  long     Use best available settings and run one 40000-iteration job.
  all      Run coarse, densify, then long.

Useful environment variables:
  SEQUENCE=Sequence_04
  SOURCE_VARIANT=partition_best
  SOURCE_DIR=/path/to/official/colmap/input
  OUT_ROOT=/path/to/output/root
  RESOLUTION=1
  RANK_BY=mean_ssim       # or mean_psnr
  OVERWRITE=1             # rerun existing experiments
USAGE
}

require_file() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
}

ensure_source_dataset() {
  if [[ -d "${SOURCE_DIR}/images" && -d "${SOURCE_DIR}/sparse/0" ]]; then
    return
  fi

  echo "Official COLMAP input not found at ${SOURCE_DIR}; preparing it from ${UNDISTORTED_DIR}"
  require_file "${UNDISTORTED_DIR}/images"
  require_file "${UNDISTORTED_DIR}/sparse"
  python "${REPO_ROOT}/scripts/prepare_official_colmap_dataset.py" \
    --undistorted-dir "${UNDISTORTED_DIR}" \
    --output-dir "${SOURCE_DIR}" \
    --overwrite
}

check_extensions() {
  python - <<'PY'
import diff_gaussian_rasterization  # noqa: F401
import simple_knn  # noqa: F401
print("Mip-Splatting CUDA extensions are importable.")
PY
}

safe_exp_name() {
  local raw="$1"
  echo "${raw}" | tr './' 'p_'
}

write_summary_row() {
  local experiment="$1"
  local iterations="$2"
  local kernel_size="$3"
  local lambda_dssim="$4"
  local densify_until="$5"
  local densify_grad="$6"
  local metrics_json="$7"

  python - "${SUMMARY_CSV}" "${experiment}" "${iterations}" "${RESOLUTION}" "${kernel_size}" "${lambda_dssim}" "${densify_until}" "${densify_grad}" "${metrics_json}" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
experiment, iterations, resolution, kernel, lam, densify_until, densify_grad, metrics_path = sys.argv[2:]
metrics = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
row = {
    "experiment": experiment,
    "iterations": iterations,
    "resolution": resolution,
    "kernel_size": kernel,
    "lambda_dssim": lam,
    "densify_until_iter": densify_until,
    "densify_grad_threshold": densify_grad,
    "image_count": metrics["image_count"],
    "mean_psnr": metrics["mean_psnr"],
    "mean_ssim": metrics["mean_ssim"],
    "min_psnr": metrics["min_psnr"],
    "max_psnr": metrics["max_psnr"],
}
summary_path.parent.mkdir(parents=True, exist_ok=True)
rows = []
if summary_path.exists():
    with summary_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
rows = [item for item in rows if item["experiment"] != experiment]
rows.append(row)
with summary_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=row.keys())
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps(row, ensure_ascii=False))
PY
}

select_best() {
  local output_env="$1"
  python - "${SUMMARY_CSV}" "${output_env}" "${RANK_BY}" <<'PY'
import csv
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
output_env = Path(sys.argv[2])
rank_by = sys.argv[3]
if not summary_path.exists():
    raise SystemExit(f"Missing summary file: {summary_path}")
with summary_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if not rows:
    raise SystemExit("No sweep rows found")
if rank_by not in rows[0]:
    raise SystemExit(f"Unknown RANK_BY={rank_by}")

def score(row):
    # Primary sort is configurable. mean_ssim is the default because the current
    # problem is blur/structure loss; mean_psnr breaks ties.
    return (float(row[rank_by]), float(row["mean_psnr"]), float(row["min_psnr"]))

best = max(rows, key=score)
output_env.write_text(
    "\n".join(
        [
            f"BEST_EXPERIMENT='{best['experiment']}'",
            f"BEST_ITERATIONS='{best['iterations']}'",
            f"BEST_RESOLUTION='{best['resolution']}'",
            f"BEST_KERNEL_SIZE='{best['kernel_size']}'",
            f"BEST_LAMBDA_DSSIM='{best['lambda_dssim']}'",
            f"BEST_DENSIFY_UNTIL='{best['densify_until_iter']}'",
            f"BEST_DENSIFY_GRAD='{best['densify_grad_threshold']}'",
            f"BEST_MEAN_PSNR='{best['mean_psnr']}'",
            f"BEST_MEAN_SSIM='{best['mean_ssim']}'",
            "",
        ]
    ),
    encoding="utf-8",
)
print("Best experiment by", rank_by)
for key in [
    "experiment",
    "iterations",
    "kernel_size",
    "lambda_dssim",
    "densify_until_iter",
    "densify_grad_threshold",
    "mean_psnr",
    "mean_ssim",
    "min_psnr",
]:
    print(f"  {key}: {best[key]}")
PY
}

run_one() {
  local iterations="$1"
  local kernel_size="$2"
  local lambda_dssim="$3"
  local densify_until="$4"
  local densify_grad="$5"

  RUN_INDEX=$((RUN_INDEX + 1))
  local kernel_tag
  local lambda_tag
  local grad_tag
  kernel_tag="$(safe_exp_name "${kernel_size}")"
  lambda_tag="$(safe_exp_name "${lambda_dssim}")"
  grad_tag="$(safe_exp_name "${densify_grad}")"
  local experiment="it${iterations}_r${RESOLUTION}_k${kernel_tag}_dssim${lambda_tag}_du${densify_until}_dg${grad_tag}"
  local exp_dir="${OUT_ROOT}/${experiment}"
  local model_dir="${exp_dir}/output"
  local eval_dir="${exp_dir}/evaluation"
  local log_dir="${exp_dir}/logs"
  local metrics_json="${eval_dir}/metrics.json"
  local port=$((BASE_PORT + RUN_INDEX))

  if [[ -f "${metrics_json}" && "${OVERWRITE}" != "1" ]]; then
    echo "Skipping existing experiment: ${experiment}"
    write_summary_row "${experiment}" "${iterations}" "${kernel_size}" "${lambda_dssim}" "${densify_until}" "${densify_grad}" "${metrics_json}"
    return
  fi

  if [[ "${OVERWRITE}" == "1" && -d "${exp_dir}" ]]; then
    case "${exp_dir}" in
      "${OUT_ROOT}"/*) rm -rf "${exp_dir}" ;;
      *) echo "Refusing to remove unexpected path: ${exp_dir}" >&2; exit 1 ;;
    esac
  fi

  mkdir -p "${model_dir}" "${eval_dir}" "${log_dir}"
  echo
  echo "=== Running ${experiment} ==="
  echo "Output: ${exp_dir}"

  python "${MIP_DIR}/train.py" \
    -s "${SOURCE_DIR}" \
    -m "${model_dir}" \
    --resolution "${RESOLUTION}" \
    --iterations "${iterations}" \
    --position_lr_max_steps "${iterations}" \
    --kernel_size "${kernel_size}" \
    --lambda_dssim "${lambda_dssim}" \
    --densify_until_iter "${densify_until}" \
    --densify_grad_threshold "${densify_grad}" \
    --test_iterations -1 \
    --save_iterations "${iterations}" \
    --port "${port}" \
    --quiet 2>&1 | tee "${log_dir}/train.log"

  python "${MIP_DIR}/render.py" \
    -m "${model_dir}" \
    -r "${RESOLUTION}" \
    --iteration "${iterations}" \
    --skip_test \
    --quiet 2>&1 | tee "${log_dir}/render.log"

  local render_dir="${model_dir}/train/ours_${iterations}/test_preds_${RESOLUTION}"
  local gt_dir="${model_dir}/train/ours_${iterations}/gt_${RESOLUTION}"
  python "${REPO_ROOT}/scripts/evaluate_image_dirs.py" \
    --renders-dir "${render_dir}" \
    --gt-dir "${gt_dir}" \
    --output-dir "${eval_dir}" 2>&1 | tee "${log_dir}/evaluate.log"

  write_summary_row "${experiment}" "${iterations}" "${kernel_size}" "${lambda_dssim}" "${densify_until}" "${densify_grad}" "${metrics_json}"
}

run_coarse() {
  echo "Running coarse sweep: kernel_size x lambda_dssim"
  local iterations="${COARSE_ITERATIONS:-30000}"
  local densify_until="${COARSE_DENSIFY_UNTIL:-20000}"
  local densify_grad="${COARSE_DENSIFY_GRAD:-0.00015}"
  for kernel_size in 0.05 0.1 0.2; do
    for lambda_dssim in 0.1 0.2 0.3; do
      run_one "${iterations}" "${kernel_size}" "${lambda_dssim}" "${densify_until}" "${densify_grad}"
    done
  done
  select_best "${OUT_ROOT}/best_coarse.env"
}

run_densify() {
  local best_env="${OUT_ROOT}/best_coarse.env"
  if [[ ! -f "${best_env}" ]]; then
    echo "Missing ${best_env}; run coarse first." >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${best_env}"
  echo "Running densify sweep from best coarse: ${BEST_EXPERIMENT}"
  local iterations="${DENSIFY_ITERATIONS:-30000}"
  for densify_until in 15000 20000; do
    for densify_grad in 0.0001 0.00015 0.0002; do
      run_one "${iterations}" "${BEST_KERNEL_SIZE}" "${BEST_LAMBDA_DSSIM}" "${densify_until}" "${densify_grad}"
    done
  done
  select_best "${OUT_ROOT}/best_densify.env"
}

run_long() {
  local best_env="${OUT_ROOT}/best_densify.env"
  if [[ ! -f "${best_env}" ]]; then
    best_env="${OUT_ROOT}/best_coarse.env"
  fi
  if [[ ! -f "${best_env}" ]]; then
    echo "Missing best env file; run coarse or densify first." >&2
    exit 1
  fi
  # shellcheck disable=SC1090
  source "${best_env}"
  echo "Running long training from best settings: ${BEST_EXPERIMENT}"
  run_one "${LONG_ITERATIONS:-40000}" "${BEST_KERNEL_SIZE}" "${BEST_LAMBDA_DSSIM}" "${BEST_DENSIFY_UNTIL}" "${BEST_DENSIFY_GRAD}"
  select_best "${OUT_ROOT}/best_long.env"
}

mkdir -p "${OUT_ROOT}"

case "${MODE}" in
  -h|--help|help)
    usage
    exit 0
    ;;
  coarse|densify|long|all)
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac

require_file "${MIP_DIR}/train.py"
require_file "${MIP_DIR}/render.py"
ensure_source_dataset
check_extensions

case "${MODE}" in
  coarse)
    run_coarse
    ;;
  densify)
    run_densify
    ;;
  long)
    run_long
    ;;
  all)
    run_coarse
    run_densify
    run_long
    ;;
esac

echo
echo "Sweep summary: ${SUMMARY_CSV}"
echo "Output root: ${OUT_ROOT}"
