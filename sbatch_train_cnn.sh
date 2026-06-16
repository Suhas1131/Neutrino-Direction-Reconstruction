#!/bin/bash

#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --gpus-per-node=1
#SBATCH --output=logs/anitaCNN_gpu.out.%j
#SBATCH --account=YOUR_ACCOUNT

module reset

# Activate your Python environment
source /path/to/your/environment/bin/activate

module load cuda/11.8.0
module load cudnn/8.7.0.84-11.8

mkdir -p logs

python -u src/train_cnn.py