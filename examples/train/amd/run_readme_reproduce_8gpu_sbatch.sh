#!/usr/bin/env bash
# Rainier 8-GPU AMD Tinker reproduction (INFERENCE_NUM_ENGINES=6).
#SBATCH --job-name=amd-rainier8
#SBATCH --partition=mi355x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:mi355x:8
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=/home/pratmish/SkyRL/logs/amd-rainier8-%j.out
#SBATCH --error=/home/pratmish/SkyRL/logs/amd-rainier8-%j.err

set -euo pipefail

export INFERENCE_NUM_ENGINES=6
export IMAGE_TAG=skyrl-amd-rocm:local

exec bash /home/pratmish/SkyRL/examples/train/amd/run_readme_reproduce_sbatch.sh
