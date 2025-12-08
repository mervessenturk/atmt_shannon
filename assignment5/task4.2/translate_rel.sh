#!/usr/bin/bash -l
#SBATCH --partition=teaching
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --gpus=1
#SBATCH --output=rel_%j.out

# Load environment
module load miniforge3
eval "$(conda shell.bash hook)"
conda activate atmt
module load cuda

# Go to project directory
cd /home/mesent/atmt_2025

# Run REL-pruning translation
python3 translate.py \
    --cuda \
    --input /home/mesent/data/atmt_2025/cz-en/data/raw/test.cz \
    --src-tokenizer /home/mesent/data/atmt_2025/cz-en/tokenizers/cz-bpe-8000.model \
    --tgt-tokenizer /home/mesent/data/atmt_2025/cz-en/tokenizers/en-bpe-8000.model \
    --checkpoint-path /home/mesent/data/atmt_2025/cz-en/checkpoints/checkpoint7_5.328.pt \
    --output /home/mesent/atmt_2025/output_rel.txt \
    --beam-size 5 \
    --max-len 128 \
    --alpha 0.7 \
    --use-rel-pruning \
    --tau 5
