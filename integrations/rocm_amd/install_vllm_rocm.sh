#!/usr/bin/env bash
# Install vLLM for ROCm without upgrading PyTorch.
# Prefer a wheel built with build_vllm_rocm.sh (cached under .vllm_rocm_cache/wheels/).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHEEL_CACHE="${SCRIPT_DIR}/.vllm_rocm_cache/wheels"

TORCH_VER="$(python3 -c 'import torch; print(torch.__version__)')"
echo "Current torch ${TORCH_VER}"

WHEEL=""
if compgen -G "${WHEEL_CACHE}/vllm-0.20*.whl" >/dev/null; then
  WHEEL="$(ls -1t "${WHEEL_CACHE}"/vllm-0.20*.whl | head -1)"
elif compgen -G "${WHEEL_CACHE}/vllm-*.whl" >/dev/null; then
  WHEEL="$(ls -1t "${WHEEL_CACHE}"/vllm-*.whl | head -1)"
fi

if [ -n "${WHEEL}" ]; then
  echo "Installing cached wheel ${WHEEL}"
  python3 -m pip install --no-cache-dir --force-reinstall --no-deps "${WHEEL}"
  python3 -m pip install --no-cache-dir -q \
    -r "${SCRIPT_DIR}/vllm_rocm_runtime_requirements.txt"
elif ! python3 "${SCRIPT_DIR}/verify_vllm_skyrl_compat.py" 2>/dev/null; then
  bash "${SCRIPT_DIR}/build_vllm_rocm.sh"
else
  echo "vLLM already installed and SkyRL-compatible"
  python3 "${SCRIPT_DIR}/verify_vllm_skyrl_compat.py"
  exit 0
fi

AFTER="$(python3 -c 'import torch; print(torch.__version__)')"
if [ "$AFTER" != "$TORCH_VER" ]; then
  echo "ERROR: torch changed ${TORCH_VER} -> ${AFTER}; refusing to continue"
  exit 1
fi

python3 "${SCRIPT_DIR}/verify_vllm_skyrl_compat.py"
python3 -c "import vllm; print('vllm', vllm.__version__, 'torch', __import__('torch').__version__)"
