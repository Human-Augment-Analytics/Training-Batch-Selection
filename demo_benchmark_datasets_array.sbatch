#!/bin/bash
#SBATCH -J bench_suite
#SBATCH -p ice-gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err
#SBATCH --array=0-7

### Attention! Set array
# array should be 1 less than total combinations, so for 20 combinations it should be 0-19

set -euo pipefail
mkdir -p logs

source ./venv/bin/activate

DATASETS=(
#  "cifar10"
#  "cifar100"
  "newt:fgvcx_plant_pathology_healthy_vs_sick"
  "newt:nabirds_species_classification_coohaw_shshaw"
#  "iwildcam"
)

STRATEGIES=(
  "random"
  "smart"
  "milo"
  "corset"
)

NUM_DATASETS=${#DATASETS[@]}
NUM_STRATEGIES=${#STRATEGIES[@]}
TOTAL=$((NUM_DATASETS * NUM_STRATEGIES))

if (( SLURM_ARRAY_TASK_ID >= TOTAL )); then
  echo "Error: array index ${SLURM_ARRAY_TASK_ID} out of range for TOTAL=$TOTAL"
  exit 1
fi

dataset_idx=$((SLURM_ARRAY_TASK_ID / NUM_STRATEGIES))
strategy_idx=$((SLURM_ARRAY_TASK_ID % NUM_STRATEGIES))

dataset="${DATASETS[$dataset_idx]}"
strategy="${STRATEGIES[$strategy_idx]}"

echo "Task ${SLURM_ARRAY_TASK_ID} / $((TOTAL-1))"
echo "dataset=$dataset"
echo "strategy=$strategy"

srun python -u -m trainer.pipelines.vision.benchmark_datasets \
  --datasets "$dataset" \
  --strategies "$strategy" \
  --epochs 10 \
  --runs 2 \
  --batch-size 64 \
  --model ResNet18 \
  --run-tag smoke-0314
