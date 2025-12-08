#!/usr/bin/bash -l
#SBATCH --job-name=avg_ckpts
#SBATCH --partition=teaching
#SBATCH --gpus=0
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=average_checkpoints_fixed.out

module load mamba
source activate atmt

export OMP_NUM_THREADS=2  # prevent OpenMP thread explosion

echo "=== Averaging checkpoints on compute node ==="
cd /home/mesent/data/atmt_2025

python /home/mesent/data/atmt_2025/average_checkpoints_fixed.py

echo "=== Done ==="
