#!/usr/bin/env bash
# Quick vLLM LLM init smoke inside a ROCm container (sync engine via Ray actor, like GRPO).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

: "${MODEL_NAME:=Qwen/Qwen2.5-0.5B-Instruct}"
export VLLM_SMOKE_GPU="${VLLM_SMOKE_GPU:-0}"
export HIP_VISIBLE_DEVICES="${VLLM_SMOKE_GPU}"
export ROCR_VISIBLE_DEVICES="${VLLM_SMOKE_GPU}"
# Ray sets CUDA_VISIBLE_DEVICES on GPU actors; required for vLLM ROCm platform detection today.
export CUDA_VISIBLE_DEVICES="${VLLM_SMOKE_GPU}"

export VLLM_TARGET_DEVICE=rocm
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export VLLM_ALLOW_INSECURE_SERIALIZATION="${VLLM_ALLOW_INSECURE_SERIALIZATION:-1}"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
export RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1

python3 - <<'PY'
import os
import ray
import torch

print("torch", torch.__version__, "hip", torch.version.hip)
print("VLLM_USE_V1", os.environ.get("VLLM_USE_V1"))
print("HIP_VISIBLE_DEVICES", os.environ.get("HIP_VISIBLE_DEVICES"))
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))

model = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

ray.init(num_gpus=1, include_dashboard=False, logging_level="error")


@ray.remote(num_gpus=1)
def _vllm_smoke(model_name: str) -> str:
  import os
  import vllm
  from vllm import LLM, SamplingParams

  print("worker vllm", vllm.__version__)
  print("worker CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))
  print("worker HIP_VISIBLE_DEVICES", os.environ.get("HIP_VISIBLE_DEVICES"))

  llm = LLM(
      model=model_name,
      tensor_parallel_size=1,
      enforce_eager=True,
      gpu_memory_utilization=0.35,
      trust_remote_code=True,
  )
  out = llm.generate(["Hello"], SamplingParams(max_tokens=8, temperature=0))
  return out[0].outputs[0].text


text = ray.get(_vllm_smoke.remote(model))
ray.shutdown()
print("generated:", text)
print("PASS: vLLM LLM smoke (Ray actor)")
PY
