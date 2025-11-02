#!/bin/bash
#SBATCH -J vision_gpu_job            # Job name
#SBATCH -N 1                        # 1 node
#SBATCH --ntasks=1                  # 1 task
#SBATCH --cpus-per-task=4           
#SBATCH --gres=gpu:1                # 1 GPU, any type
#SBATCH --mem=16G                   # modest RAM
#SBATCH -t 04:00:00                 # walltime hh:mm:ss
#SBATCH -o logs/%x-%j.out           # stdout
#SBATCH -e logs/%x-%j.err           # stderr

##SBATCH --mail-type=BEGIN,END,FAIL		# uncomment if you want emails 
##SBATCH --mail-user=<username>@gatech.edu	# uncomment if you want emails

set -euo pipefail
mkdir -p logs

echo "Node(s): $SLURM_NODELIST"
echo "CUDA devices visible: $CUDA_VISIBLE_DEVICES"
nvidia-smi || true

# activate Python env
source ./venv/bin/activate

# load modules if the env needs them
# module load cuda
# module load gcc

# run job
srun -u python -m trainer.pipelines.vision.benchmark_datasets
