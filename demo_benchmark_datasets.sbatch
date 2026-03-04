#!/bin/bash
#SBATCH -J vision_gpu_job            # Job name
#SBATCH -N 1                        # 1 node
#SBATCH --ntasks=1                  # 1 task
#SBATCH --cpus-per-task=4           
#SBATCH --gres=gpu:H100:1           # --gres=gpu:1                # 1 GPU, any type   
#SBATCH --mem-per-gpu=224GB         # --mem=16G                   # modest RAM        
#SBATCH -t 12:00:00                 # walltime hh:mm:ss
#SBATCH -o logs/%x-%j.out           # stdout
#SBATCH -e logs/%x-%j.err           # stderr

##SBATCH --mail-type=BEGIN,END,FAIL		# uncomment if you want emails 
##SBATCH --mail-user=<username>@gatech.edu	# uncomment if you want emails

set -euo pipefail
mkdir -p logs

echo "Node(s): $SLURM_NODELIST"
echo "CUDA devices visible: ${CUDA_VISIBLE_DEVICES-<none>}"

echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS-<none>}"
echo "CUDA_VISIBLE_DEVICES raw: '${CUDA_VISIBLE_DEVICES-<none>}'"

echo "nvidia-smi:"
nvidia-smi || true

python -u - <<'PY'
import os, torch
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

# activate Python env
source ./venv/bin/activate

# load modules if the env needs them
#module load cuda
#module load gcc

echo "GPU info:"
python -u -c "import torch;
print('cuda available:', torch.cuda.is_available());
print('device count:', torch.cuda.device_count());
print('device 0:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# run job
echo "Running benchmarks:"
python -u -m trainer.pipelines.vision.benchmark_datasets
