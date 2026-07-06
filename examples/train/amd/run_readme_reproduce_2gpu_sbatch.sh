#!/usr/bin/env bash
# Rainier 2-GPU AMD Tinker reproduction (INFERENCE_NUM_ENGINES=1).
#SBATCH --job-name=amd-rainier2
#SBATCH --partition=mi355x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:mi355x:2
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/home/pratmish/SkyRL/logs/amd-rainier2-%j.out
#SBATCH --error=/home/pratmish/SkyRL/logs/amd-rainier2-%j.err

set -euo pipefail

export INFERENCE_NUM_ENGINES=1
export IMAGE_TAG=skyrl-amd-rocm:local

exec bash /home/pratmish/SkyRL/examples/train/amd/run_readme_reproduce_sbatch.sh
