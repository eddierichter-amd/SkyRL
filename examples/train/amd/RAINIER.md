# Rainier (MI355x) Cluster Notes

This document captures **Rainier-specific** settings validated on AMD MI355x nodes
(`mi355x-pollara-*`, `mi355x-thor-*`, `mi355x-dlc-pollara-*`) for the single-node
AMD Tinker workflow in this directory.

The upstream README defaults target a generic 8-GPU AMD node. On Rainier, use the
overrides below when FSDP training and vLLM inference are split across GPUs
(`colocate_all=false`, the default in `run_tinker_server_amd.sh`).

## Quick start (SLURM)

From the repository root on `rainier-login`:

```bash
# 2-GPU smoke (INFERENCE_NUM_ENGINES=1)
sbatch examples/train/amd/run_readme_reproduce_2gpu_sbatch.sh

# 8-GPU README-default layout (INFERENCE_NUM_ENGINES=6)
sbatch examples/train/amd/run_readme_reproduce_8gpu_sbatch.sh
```

Logs land under `logs/` relative to the repo checkout.

## Rainier runtime environment

Source before starting the Tinker server (inside the container or on the host):

```bash
source examples/train/amd/run_rainier_env.sh
```

Or export manually:

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_V1=1
export INFERENCE_DISTRIBUTED_EXECUTOR_BACKEND=mp
export INFERENCE_MAX_MODEL_LEN=4096
export INFERENCE_MAX_NUM_BATCHED_TOKENS=8192
export INFERENCE_MAX_NUM_SEQS=64
export INFERENCE_GPU_MEMORY_UTILIZATION=0.5
```

### Why these are needed on Rainier

| Variable | README default | Rainier value | Reason |
|----------|----------------|---------------|--------|
| `VLLM_ENABLE_V1_MULTIPROCESSING` | (unset) | `0` | vLLM V1 worker spawn can fail after FSDP initializes CUDA on ROCm |
| `VLLM_USE_V1` | (unset) | `1` | Keep the vLLM V1 engine path |
| `INFERENCE_DISTRIBUTED_EXECUTOR_BACKEND` | `ray` | `mp` | More reliable vLLM startup when training and inference are on separate GPUs |
| `INFERENCE_MAX_MODEL_LEN` | `32768` | `4096` | Avoid OOM / engine init failures on MI355x |
| `INFERENCE_MAX_NUM_BATCHED_TOKENS` | `32768` | `8192` | Same |
| `INFERENCE_MAX_NUM_SEQS` | `256` | `64` | Same |
| `INFERENCE_GPU_MEMORY_UTILIZATION` | `0.8` | `0.5` | Same |

SkyRL also sets `VLLM_ENABLE_V1_MULTIPROCESSING=0` in the Ray runtime env via
`skyrl/train/utils/utils.py` when `VLLM_USE_V1` is unset. Setting it explicitly
on Rainier avoids edge cases where the variable is already set without the
multiprocessing disable.

## GPU layout

| Node size | `INFERENCE_NUM_ENGINES` | Notes |
|-----------|-------------------------|-------|
| 2 GPUs | `1` | 1 policy + 1 inference |
| 8 GPUs | `6` | README default: 1 policy + 6 inference + headroom |

## Docker on Rainier

Build once on a compute node (or reuse a cached image):

```bash
docker build -f docker/Dockerfile.amd -t skyrl-amd-rocm:local .
```

Run with ROCm devices and host networking (same as README):

```bash
docker run --rm -it \
  --network host \
  --ipc=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -e HOME=/tmp \
  -v "$(pwd):/workspace/SkyRL" \
  skyrl-amd-rocm:local
```

Inside the container:

```bash
cd /workspace/SkyRL/examples/train/amd
source run_rainier_env.sh
INFERENCE_NUM_ENGINES=1 bash run_tinker_server_amd.sh
```

## Client workflow

### Hello-world

```bash
TINKER_BASE_URL=http://localhost:9000 TINKER_API_KEY=tml-dummy \
python tinker_hello_world.py
```

### GRPO — restart the server first

On the FSDP backend, only **one LoRA adapter per worker group** is supported.
After `tinker_hello_world.py`, restart the Tinker server before running GRPO:

```bash
# stop the old server, then:
source run_rainier_env.sh
INFERENCE_NUM_ENGINES=1 bash run_tinker_server_amd.sh
```

Then in a second shell:

```bash
TINKER_BASE_URL=http://localhost:9000 TINKER_API_KEY=tml-dummy \
python grpo_client.py \
  --max-train-steps 1 \
  --num-prompts 2 \
  --group-size 2 \
  --max-tokens 64 \
  --max-train-examples 32 \
  --max-val-examples 8 \
  --reprepare-data
```

Running GRPO on the same server immediately after hello-world fails with
`register_adapter is not implemented` on FSDP.

## Validated nodes (Jul 2026)

| Job | Node | GPUs | Result |
|-----|------|------|--------|
| README 2-GPU repro | `mi355x-pollara-1` | 2 | PASS (with Rainier overrides) |
| README 8-GPU repro | `mi355x-thor-4` | 8 | PASS (with Rainier overrides) |
| Literal README (no overrides) | `mi355x-dlc-pollara-3` | 2 | FAIL — vLLM engine init / OOM |

## Partition and resources

- **Partition:** `mi355x`
- **GRES:** `gpu:mi355x:N` (e.g. `--gres=gpu:mi355x:2`)
- Typical single-node request: `--cpus-per-task=16` (2 GPU) or `--cpus-per-task=64` (8 GPU)
