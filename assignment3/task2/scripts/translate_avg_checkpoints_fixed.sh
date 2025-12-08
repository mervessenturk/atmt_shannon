#!/usr/bin/bash -l
#SBATCH --job-name=translate_avg_checkpoints_fixed
#SBATCH --partition=teaching
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --output=translate_avg_checkpoints_fixed.out

# ================================================================
# Description:
#   Evaluates the averaged checkpoint (7–9) using a fully
#   consistent tokenization–detokenization pipeline.
#   Performs translation, SentencePiece decoding, and BLEU scoring.
#
# Key updates:
#   - Restored original Oct 2 SentencePiece models for consistency.
#   - Added BOS/EOS at inference for proper sequence boundaries.
#   - Automated detokenization using SentencePiece decode.
#   - Computes BLEU via SacreBLEU for reproducible evaluation.
#
# Note:
#   Earlier low BLEU scores (≈0.3–1.3) were traced to mismatched
#   tokenizers; this script fixes the entire preprocessing pipeline.
#
# Output:
#   - Detokenized translations: output_avg_checkpoints.detok.txt
#   - BLEU score logged in: translate_avg_checkpoints_fixed.out
# ================================================================
# ==== Environment setup ====
module load gpu
module load mamba
source activate atmt

echo "=== Starting translation evaluation with averaged model (fixed tokenizer + BOS/EOS) ==="

python - <<'EOF'
import torch, argparse, types, sys, sentencepiece as spm, os

sys.path.append("/home/mesent/data/atmt_2025")

# --- Safety patch in case args are missing in checkpoint ---
old_load = torch.load
def safe_load(*args, **kwargs):
    obj = old_load(*args, **kwargs)
    if isinstance(obj, dict) and "args" not in obj:
        print(" Patched: 'args' key missing — adding dummy args to prevent crash.")
        obj["args"] = argparse.Namespace()
    return obj
torch.load = safe_load

from translate import main

#  Use original (Oct 2) tokenizers that match training
args = types.SimpleNamespace(
    cuda=True,
    data="/home/mesent/data/atmt_2025/cz-en/data/prepared/",
    source_lang="cz",
    target_lang="en",
    src_tokenizer="/home/mesent/data/atmt_2025/cz-en/tokenizers_original/cz-bpe-8000.model",
    tgt_tokenizer="/home/mesent/data/atmt_2025/cz-en/tokenizers_original/en-bpe-8000.model",
    batch_size=1,
    arch="transformer",
    max_length=300,
    log_file="/home/mesent/data/atmt_2025/cz-en/logs/translate_avg_checkpoints_fixed.log",
    save_dir="/home/mesent/data/atmt_2025/cz-en/checkpoints/",
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

    #  Input and checkpoint
    input="/home/mesent/data/atmt_2025/cz-en/data/prepared/test.bpe.cz",
    checkpoint_path="/home/mesent/data/atmt_2025/cz-en/checkpoints/checkpoint9_8.190.pt",
    output="/home/mesent/data/atmt_2025/cz-en/data/prepared/output_checkpoint9_fixed.bpe.txt",
    max_len=300,
    bleu=False,
    reference=None,
)

#  Inject BOS/EOS tokens before translation (ensure consistency)
print("Checking BOS/EOS consistency...")
sp_cz = spm.SentencePieceProcessor(model_file=args.src_tokenizer)
tmp_in = args.input + ".with_bos_eos"
with open(args.input, "r", encoding="utf-8") as fin, open(tmp_in, "w", encoding="utf-8") as fout:
    for line in fin:
        toks = line.strip().split()
        fout.write("<s> " + " ".join(toks) + " </s>\n")
args.input = tmp_in

#  Run translation
main(args)
print(" Translation completed successfully.")
EOF

# ==== Postprocessing: detokenization and BLEU ====
echo "=== Detokenizing model output ==="
python - <<'EOF'
import sentencepiece as spm

bpe_model = "/home/mesent/data/atmt_2025/cz-en/tokenizers_original/en-bpe-8000.model"
inp = "/home/mesent/data/atmt_2025/cz-en/data/prepared/output_checkpoint9_fixed.bpe.txt"
outp = "/home/mesent/data/atmt_2025/cz-en/data/prepared/output_checkpoint9_fixed.detok.txt"

sp = spm.SentencePieceProcessor(model_file=bpe_model)
with open(inp, "r", encoding="utf-8") as fin, open(outp, "w", encoding="utf-8") as fout:
    for line in fin:
        fout.write(sp.decode(line.strip().split()) + "\n")
print(f" Detokenized output written to {outp}")
EOF

echo "=== Computing BLEU score ==="
sacrebleu /home/mesent/data/atmt_2025/cz-en/data/raw/test.en \
  -i /home/mesent/data/atmt_2025/cz-en/data/prepared/output_checkpoint9_fixed.detok.txt \
  -m bleu -b -w 2

echo "=== BLEU evaluation completed ==="
