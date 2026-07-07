#!/usr/bin/env bash
set -euo pipefail

# Create the conda environments needed for the lunar FM benchmark runs.
#
# Why multiple environments?
# MACE-MH currently needs mace-torch/e3nn versions that conflict with the
# FAIRChem/UMA and NequIP-OAM stack. Keeping model families in separate envs is
# the reproducible route used for the benchmark in this repository.
#
# Usage:
#   bash setup/install_model_envs.sh
#
# Useful overrides:
#   MODELS="sevennet mattersim" bash setup/install_model_envs.sh
#   TORCH_CUDA_OLD=cu118 TORCH_CUDA_FM=cu121 bash setup/install_model_envs.sh
#   ENV_PREFIX=/path/to/envs bash setup/install_model_envs.sh

MODELS="${MODELS:-sevennet mattersim upet mace_mh fm_latest}"
ENV_PREFIX="${ENV_PREFIX:-}"
PYTHON_OLD="${PYTHON_OLD:-3.10}"
PYTHON_FM="${PYTHON_FM:-3.11}"
TORCH_CUDA_OLD="${TORCH_CUDA_OLD:-cu121}"
TORCH_CUDA_FM="${TORCH_CUDA_FM:-cu128}"
INSTALL_KERNELS="${INSTALL_KERNELS:-1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Load Anaconda/Miniconda/Mambaforge first." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

env_arg() {
  local name="$1"
  if [[ -n "${ENV_PREFIX}" ]]; then
    mkdir -p "${ENV_PREFIX}"
    echo "--prefix ${ENV_PREFIX}/${name}"
  else
    echo "--name ${name}"
  fi
}

activate_env() {
  local name="$1"
  if [[ -n "${ENV_PREFIX}" ]]; then
    conda activate "${ENV_PREFIX}/${name}"
  else
    conda activate "${name}"
  fi
}

create_env() {
  local name="$1"
  local py="$2"
  if [[ -n "${ENV_PREFIX}" && -d "${ENV_PREFIX}/${name}" ]]; then
    echo "Environment ${ENV_PREFIX}/${name} already exists; reusing it."
  elif [[ -z "${ENV_PREFIX}" ]] && conda env list | awk '{print $1}' | grep -qx "${name}"; then
    echo "Environment ${name} already exists; reusing it."
  else
    # shellcheck disable=SC2046
    conda create -y $(env_arg "${name}") "python=${py}" pip
  fi
  activate_env "${name}"
  python -m pip install --upgrade pip setuptools wheel
}

install_common() {
  python -m pip install \
    ase \
    numpy \
    pandas \
    scipy \
    matplotlib \
    nbformat \
    nbclient \
    ipykernel \
    psutil \
    tqdm
}

install_torch() {
  local cuda_tag="$1"
  python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${cuda_tag}"
}

register_kernel() {
  local env_name="$1"
  local display="$2"
  if [[ "${INSTALL_KERNELS}" == "1" ]]; then
    python -m ipykernel install --user --name "${env_name}" --display-name "${display}"
  fi
}

install_sevennet() {
  create_env sevennet_env "${PYTHON_OLD}"
  install_torch "${TORCH_CUDA_OLD}"
  install_common
  python -m pip install "sevenn==0.12.1"
  register_kernel sevennet_env "sevennet_env"
}

install_mattersim() {
  create_env mattersim_env "${PYTHON_OLD}"
  install_torch "${TORCH_CUDA_OLD}"
  install_common
  python -m pip install "mattersim==1.2.1"
  register_kernel mattersim_env "mattersim_env"
}

install_upet() {
  create_env upet_env "${PYTHON_OLD}"
  install_torch "${TORCH_CUDA_OLD}"
  install_common
  python -m pip install "upet==0.1.2"
  register_kernel upet_env "upet_env"
}

install_mace_mh() {
  create_env mace_latest_env "${PYTHON_OLD}"
  install_torch "${TORCH_CUDA_OLD}"
  install_common
  python -m pip install "mace-torch==0.3.16"
  register_kernel mace_latest_env "mace_latest_env"
}

install_fm_latest() {
  create_env fm_latest_models "${PYTHON_FM}"
  install_torch "${TORCH_CUDA_FM}"
  install_common
  python -m pip install "fairchem-core==2.20.0" "nequip==0.17.1"
  register_kernel fm_latest_models "fm_latest_models"
}

for model in ${MODELS}; do
  case "${model}" in
    sevennet) install_sevennet ;;
    mattersim) install_mattersim ;;
    upet) install_upet ;;
    mace_mh|mace) install_mace_mh ;;
    fm_latest|uma|nequip_oam_l) install_fm_latest ;;
    *)
      echo "Unknown model/env key: ${model}" >&2
      echo "Allowed: sevennet mattersim upet mace_mh fm_latest" >&2
      exit 1
      ;;
  esac
done

echo
echo "Environment installation complete."
echo "For UMA, set your Hugging Face token before running:"
echo "  export HF_TOKEN=<your_token>"
echo "For NequIP-OAM-L, copy or download the compiled model to:"
echo "  models/nequip/mir-group__NequIP-OAM-L__0.1.nequip.pth"
