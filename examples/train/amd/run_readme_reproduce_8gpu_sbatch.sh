#!/usr/bin/env bash
# README reproduction with default INFERENCE_NUM_ENGINES=6 (requires 8 GPUs).
#SBATCH --job-name=amd-readme8
#SBATCH --partition=mi355x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:mi355x:8
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=logs/amd-readme8-%j.out
#SBATCH --error=logs/amd-readme8-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export INFERENCE_NUM_ENGINES=6
export IMAGE_TAG=skyrl-amd-rocm:local

exec bash "${SCRIPT_DIR}/run_readme_reproduce_sbatch.sh"
