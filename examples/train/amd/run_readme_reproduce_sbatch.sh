#!/usr/bin/env bash
# End-to-end AMD README workflow with Rainier MI355x overrides.
#SBATCH --job-name=amd-rainier
#SBATCH --partition=mi355x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:mi355x:8
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=/home/pratmish/SkyRL/logs/amd-rainier-%j.out
#SBATCH --error=/home/pratmish/SkyRL/logs/amd-rainier-%j.err

set -euo pipefail

REPO_ROOT="/home/pratmish/SkyRL"
cd "${REPO_ROOT}"
mkdir -p logs

IMAGE_TAG="${IMAGE_TAG:-skyrl-amd-rocm:local}"
if ! docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  if docker image inspect skyrl-amd-rocm:latest >/dev/null 2>&1; then
    docker tag skyrl-amd-rocm:latest "${IMAGE_TAG}"
  fi
fi
JOB_TAG="${SLURM_JOB_ID:-local}"
CONTAINER="amd-rainier-${JOB_TAG}"
AMD_DIR="/workspace/SkyRL/examples/train/amd"

# Rainier overrides (see RAINIER.md).
source "${REPO_ROOT}/examples/train/amd/run_rainier_env.sh"
export INFERENCE_NUM_ENGINES="${INFERENCE_NUM_ENGINES:-6}"

DOCKER_COMMON=(
  --network host
  --ipc=host
  --device=/dev/kfd
  --device=/dev/dri
  --group-add video
  --cap-add=SYS_PTRACE
  --security-opt seccomp=unconfined
  -e HOME=/tmp
  -e HF_HOME=/tmp/hf_cache
  -e SKYRL_RAY_NUM_CPUS=64
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0
  -e VLLM_USE_V1=1
  -v "${REPO_ROOT}:/workspace/SkyRL"
)

step() {
  echo
  echo "================================================================"
  echo "STEP: $*"
  echo "================================================================"
}

failures=0
record() {
  local name="$1" status="$2"
  if [[ "${status}" -eq 0 ]]; then
    echo ">>> ${name}: PASS"
  else
    echo ">>> ${name}: FAIL (exit ${status})"
    failures=$((failures + 1))
  fi
}

start_server() {
  local label="$1"
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
  rm -rf "${REPO_ROOT}/skyrl/tinker/tinker.db"* 2>/dev/null || true

  docker run -d \
    --name "${CONTAINER}" \
    "${DOCKER_COMMON[@]}" \
    -w "${AMD_DIR}" \
    "${IMAGE_TAG}" \
    bash -lc "
      rm -rf /tmp/skyrl* /tmp/ray /tmp/tinker.db* 2>/dev/null || true
      source run_rainier_env.sh
      export INFERENCE_NUM_ENGINES=${INFERENCE_NUM_ENGINES}
      bash run_tinker_server_amd.sh
    "

  for _ in $(seq 1 120); do
    if curl -sf http://localhost:9000/api/v1/healthz >/dev/null 2>&1; then
      echo "${label}: server healthy on :9000"
      return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER}"; then
      echo "${label}: container exited early"
      docker logs "${CONTAINER}" | tail -80 || true
      return 1
    fi
    sleep 10
  done
  echo "${label}: health check timeout"
  docker logs "${CONTAINER}" | tail -80 || true
  return 1
}

run_client() {
  docker exec "${CONTAINER}" bash -lc "$1"
}

cleanup() {
  docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Rainier AMD reproduction job ${JOB_TAG} on $(hostname)"
echo "git=$(git rev-parse --short HEAD)"
echo "date=$(date -Is)"

step "1. Build image"
if docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1; then
  echo "Using cached image ${IMAGE_TAG}"
  record "docker build" 0
else
  set +e
  build_log="logs/amd-rainier-${JOB_TAG}-build.log"
  docker build -f docker/Dockerfile.amd -t "${IMAGE_TAG}" . >"${build_log}" 2>&1
  build_status=$?
  cat "${build_log}"
  set -e
  record "docker build" "${build_status}"
  [[ "${build_status}" -eq 0 ]] || exit "${build_status}"
fi

step "2. Container smoke"
docker run --rm "${DOCKER_COMMON[@]}" "${IMAGE_TAG}" bash -lc \
  'python -c "import ray,torch,vllm; print(\"ray\", ray.__version__); print(\"torch\", torch.__version__); print(\"vllm\", vllm.__version__)"'
record "container smoke" $?

step "3. Start Tinker server"
set +e
start_server "Rainier server"
server_status=$?
set -e
record "start tinker server" "${server_status}"
[[ "${server_status}" -eq 0 ]] || exit 1

step "4. Hello-world"
set +e
run_client "
  cd ${AMD_DIR}
  TINKER_BASE_URL=http://localhost:9000 TINKER_API_KEY=tml-dummy python tinker_hello_world.py
"
hello_status=$?
set -e
record "tinker_hello_world.py" "${hello_status}"

step "5. GRPO on fresh server (FSDP requires restart)"
set +e
start_server "GRPO server"
grpo_server_status=$?
if [[ "${grpo_server_status}" -eq 0 ]]; then
  run_client "
    cd ${AMD_DIR}
    TINKER_BASE_URL=http://localhost:9000 TINKER_API_KEY=tml-dummy python grpo_client.py \
      --max-train-steps 1 \
      --num-prompts 2 \
      --group-size 2 \
      --max-tokens 64 \
      --max-train-examples 32 \
      --max-val-examples 8 \
      --reprepare-data
  "
  grpo_status=$?
else
  grpo_status=${grpo_server_status}
fi
set -e
record "grpo_client.py (fresh server)" "${grpo_status}"

echo
echo "================================================================"
echo "SUMMARY"
echo "================================================================"
echo "hello-world: ${hello_status}"
echo "grpo fresh: ${grpo_status}"
echo "failures recorded: ${failures}"

if [[ "${hello_status}" -eq 0 && "${grpo_status}" -eq 0 ]]; then
  echo "OVERALL: PASS"
  exit 0
fi
exit 1
