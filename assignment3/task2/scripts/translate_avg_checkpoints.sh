#!/usr/bin/bash -l
#SBATCH --job-name=translate_avg_checkpoints
#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=translate_avg_checkpoints.out

# ==== Environment setup ====
module load gpu
module load mamba
source activate atmt

echo "=== Starting translation evaluation with averaged model ==="

python - <<'EOF'
import torch, argparse, types, sys
sys.path.append("/home/mesent/data/atmt_2025")

# --- Patch torch.load so that missing 'args' doesn't crash ---
old_load = torch.load
def safe_load(*args, **kwargs):
    obj = old_load(*args, **kwargs)
    if isinstance(obj, dict) and "args" not in obj:
        print("⚠️ Patched: 'args' key missing — adding dummy args to prevent crash.")
        obj["args"] = argparse.Namespace()
    return obj
torch.load = safe_load

from translate import main

args = types.SimpleNamespace(
    cuda=True,
    data="cz-en/data/prepared/",
    source_lang="cz",
    target_lang="en",
    src_tokenizer="/home/mesent/data/atmt_2025/cz-en/tokenizers/cz-bpe-8000.model",
    tgt_tokenizer="/home/mesent/data/atmt_2025/cz-en/tokenizers/en-bpe-8000.model",
    batch_size=1,
    arch="transformer",
    max_length=300,
    log_file="cz-en/logs/translate_avg_checkpoints.log",
    save_dir="cz-en/checkpoints/",
    encoder_dropout=0.1,
    decoder_dropout=0.1,
    dim_embedding=256,
    attention_heads=4,
    dim_feedforward_encoder=1024,
    dim_feedforward_decoder=1024,
    max_seq_len=300,
    n_encoder_layers=3,
    n_decoder_layers=3,
    seed=42,
    input="/home/mesent/data/atmt_2025/cz-en/data/raw/test.cz",
    checkpoint_path="/home/mesent/data/atmt_2025/cz-en/checkpoints/checkpoint_averaged_7-9.pt",
    output="/home/mesent/data/atmt_2025/output_avg_checkpoints.txt",
    max_len=300,
    bleu=False,
    reference=None,
)

main(args)
EOF

echo "=== Translation completed ==="
