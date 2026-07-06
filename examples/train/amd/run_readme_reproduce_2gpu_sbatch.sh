#!/usr/bin/env bash
# README reproduction on 2 GPUs (Rainier smoke when 8 GPUs unavailable).
# Uses INFERENCE_NUM_ENGINES=1 instead of README default 6.
#SBATCH --job-name=amd-readme2
#SBATCH --partition=mi355x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:mi355x:2
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=logs/amd-readme2-%j.out
#SBATCH --error=logs/amd-readme2-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export INFERENCE_NUM_ENGINES=1
export IMAGE_TAG=skyrl-amd-rocm:local

exec bash "${SCRIPT_DIR}/run_readme_reproduce_sbatch.sh"
