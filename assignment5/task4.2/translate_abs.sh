#!/usr/bin/bash -l
#SBATCH --partition teaching
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --gpus=1
#SBATCH --output=abs_%j.out

# Load modules
module load miniforge3
eval "$(conda shell.bash hook)" 
conda activate atmt
module load cuda

export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/pkgs/cuda-toolkit

# Go to project directory
cd /home/mesent/atmt_2025

python3 translate.py \
    --cuda \
    --input /home/mesent/data/atmt_2025/cz-en/data/raw/test.cz \
    --src-tokenizer /home/mesent/data/atmt_2025/cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer /home/mesent/data/atmt_2025/cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path /home/mesent/data/atmt_2025/cz-en/checkpoints/checkpoint7_5.328.pt \
    --output /home/mesent/atmt_2025/output_abs.txt \
    --beam-size 5 \
    --max-len 128 \
    --alpha 0.7 \
    --use-abs-pruning \
    --abs-threshold -200

