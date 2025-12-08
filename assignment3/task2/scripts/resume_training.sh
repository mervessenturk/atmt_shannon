#!/usr/bin/bash -l
#SBATCH --job-name=resume_train
#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=resume_train.out

# ==== Environment setup ====
module load gpu
module load mamba
source activate atmt

# ==== System memory snapshot ====
echo "=== Memory status before training ===" >> resume_train.out
free -h >> resume_train.out
echo "=====================================" >> resume_train.out

# ==== Training ====
python /home/mesent/data/atmt_2025/train.py --cuda \
  --data /home/mesent/data/atmt_2025/cz-en/data/prepared \
  --src-tokenizer /home/mesent/data/atmt_2025/cz-en/tokenizers/cz-bpe-8000.model \
  --tgt-tokenizer /home/mesent/data/atmt_2025/cz-en/tokenizers/en-bpe-8000.model \
  --restore-file /home/mesent/data/atmt_2025/cz-en/checkpoints/checkpoint_last.pt \
  --save-dir /home/mesent/data/atmt_2025/cz-en/checkpoints \
  --max-epoch 10 \
  --save-interval 1 \
  --epoch-checkpoints \
  --log-file /home/mesent/data/atmt_2025/cz-en/logs/resume_train.log \
  --source-lang cz \
  --target-lang en \
  --dim-embedding 256 \
  --attention-heads 4 \
  --dim-feedforward-encoder 1024 \
  --dim-feedforward-decoder 1024 \
  --n-encoder-layers 3 \
  --n-decoder-layers 3 \
  --max-seq-len 300

# ==== Memory status after training ====
echo "=== Memory status after training ===" >> resume_train.out
free -h >> resume_train.out
echo "=====================================" >> resume_train.out
