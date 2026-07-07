#!/usr/bin/env bash
set -euo pipefail

STRUCTURE_DIR="${STRUCTURE_DIR:-structures/balanced_xyz}"
STEPS="${STEPS:-1000}"
DEVICE="${DEVICE:-cuda}"
OUT_ROOT="${OUT_ROOT:-results_bulk}"
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
  for structure in "${STRUCTURE_DIR}"/*_balanced.xyz; do
    echo "Running ${model} on ${structure}"
    case "${model}" in
      sevennet|mattersim|upet)
        python scripts/bulk_md/run_benchmark.py \
          --model "${model}" \
          --structure "${structure}"
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
