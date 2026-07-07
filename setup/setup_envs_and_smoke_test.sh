#!/usr/bin/env bash
set -euo pipefail

# Create all conda environments needed for the six-model benchmark and run
# import-level smoke checks.
#
# Usage from the project root:
#   bash setup/setup_envs_and_smoke_test.sh
#
# Useful overrides:
#   ENV_PREFIX=/path/to/envs bash setup/setup_envs_and_smoke_test.sh
#   SKIP_INSTALL=1 bash setup/setup_envs_and_smoke_test.sh
#   MODELS="sevennet mace_mh uma" bash setup/setup_envs_and_smoke_test.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

INSTALL_SCRIPT="${ROOT}/setup/install_model_envs.sh"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
MODELS="${MODELS:-sevennet mattersim upet mace_mh uma nequip_oam_l}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda was not found. Load Anaconda/Miniconda/Mambaforge first." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

activate_env() {
  local env_name="$1"
  if [[ -n "${ENV_PREFIX:-}" ]]; then
    conda activate "${ENV_PREFIX}/${env_name}"
  else
    conda activate "${env_name}"
  fi
}

env_for_model() {
  case "$1" in
    sevennet) echo "sevennet_env" ;;
    mattersim) echo "mattersim_env" ;;
    upet) echo "upet_env" ;;
    mace_mh) echo "mace_latest_env" ;;
    uma|nequip_oam_l) echo "fm_latest_models" ;;
    *) echo "Unknown model: $1" >&2; exit 1 ;;
  esac
}

import_probe() {
  local model="$1"
  case "${model}" in
    sevennet) python -c "import torch, ase, sevenn; print('import OK: sevennet')" ;;
    mattersim) python -c "import torch, ase, mattersim; print('import OK: mattersim')" ;;
    upet) python -c "import torch, ase, upet; print('import OK: upet')" ;;
    mace_mh) python -c "import torch, ase, mace; print('import OK: mace_mh')" ;;
    uma) python -c "import torch, ase, fairchem; print('import OK: uma/fairchem')" ;;
    nequip_oam_l) python -c "import torch, ase, nequip; print('import OK: nequip_oam_l')" ;;
  esac
}

echo "Project root: ${ROOT}"

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  echo
  echo "=== Installing/reusing conda environments ==="
  MODELS="sevennet mattersim upet mace_mh fm_latest" bash "${INSTALL_SCRIPT}"
else
  echo
  echo "SKIP_INSTALL=1; assuming conda environments already exist."
fi

echo
echo "=== Running import smoke checks ==="
for model in ${MODELS}; do
  env_name="$(env_for_model "${model}")"
  echo
  echo "=== ${model}: activating ${env_name} ==="
  activate_env "${env_name}"
  import_probe "${model}"
done

echo
echo "All requested environment checks completed."
