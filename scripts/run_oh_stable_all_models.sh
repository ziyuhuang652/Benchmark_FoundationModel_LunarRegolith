#!/usr/bin/env bash
set -euo pipefail

STEPS="${STEPS:-2000}"
STRUCTURE_DIR="${STRUCTURE_DIR:-structures/hydroxylated_stable}"
DEVICE="${DEVICE:-cuda}"
OUT_ROOT="${OUT_ROOT:-results_oh_stable}"
ANALYSIS_DIR="${ANALYSIS_DIR:-analysis_oh_stable}"
FIGURES_DIR="${FIGURES_DIR:-figures_oh_stable}"
MODELS="${MODELS:-sevennet mattersim upet mace_mh1 uma nequip_oam_l}"
NEQUIP_MODEL="${NEQUIP_MODEL:-models/nequip/mir-group__NequIP-OAM-L__0.1.nequip.pth}"

activate_model_env() {
  local model="$1"
  if ! command -v conda >/dev/null 2>&1; then
    return
  fi
  source "$(conda info --base)/etc/profile.d/conda.sh"
  case "${model}" in
    sevennet) conda activate sevennet_env ;;
    mattersim) conda activate mattersim_env ;;
    upet) conda activate upet_env ;;
    mace_mh1) conda activate mace_latest_env ;;
    uma|nequip_oam_l) conda activate fm_latest_models ;;
    *) echo "Unknown model: ${model}" >&2; exit 1 ;;
  esac
}

for model in ${MODELS}; do
  activate_model_env "${model}"
  for structure in "${STRUCTURE_DIR}"/*_stable_surface_NBO_H.xyz; do
    echo "Running ${model} on ${structure}"
    case "${model}" in
      sevennet|mattersim|upet)
        python scripts/oh_surface/run_benchmark.py \
          --model "${model}" \
          --structure "${structure}" \
          --steps "${STEPS}" \
          --device "${DEVICE}" \
          --out-root "${OUT_ROOT}"
        ;;
      mace_mh1|uma)
        python scripts/bulk_md/run_extended_model_benchmark.py \
          --model "${model}" \
          --structure "${structure}" \
          --steps "${STEPS}" \
          --device "${DEVICE}" \
          --out-root "${OUT_ROOT}"
        ;;
      nequip_oam_l)
        python scripts/bulk_md/run_extended_model_benchmark.py \
          --model "${model}" \
          --structure "${structure}" \
          --steps "${STEPS}" \
          --device "${DEVICE}" \
          --out-root "${OUT_ROOT}" \
          --nequip-model "${NEQUIP_MODEL}"
        ;;
    esac
  done
done

python scripts/oh_surface/analyze_oh_trajectories.py "${OUT_ROOT}" \
  --metadata structures/hydroxylated_stable/oh_site_metadata.json \
  --analysis-dir "${ANALYSIS_DIR}" \
  --figures-dir "${FIGURES_DIR}" \
  --last-n-frames 1000
