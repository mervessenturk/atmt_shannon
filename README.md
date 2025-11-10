# ATMT Assignment 3 — Group Shannon
## Task 1 
## Task 2 — Checkpoint Averaging and Regularization

This task extends the baseline Transformer model from Task 1 by implementing **checkpoint averaging** and enforcing **tokenization consistency**. The goal was to stabilize translation quality and improve BLEU scores by averaging the final model weights across several epochs.

---

### Experiment Setup

**Base configuration (same as Task 1):**
- Transformer with 3 encoder / 3 decoder layers  
- Embedding dimension: 256  
- Feedforward dimension: 1024  
- Attention heads: 4  
- Dropout: 0.1  
- Maximum sequence length: 300  
- Optimizer: Adam (lr=0.0003)  
- Hardware: UZH Euler HPC (1× NVIDIA A100 GPU, 64 GB memory)

**Improvements introduced:**
1. Implemented checkpoint averaging over epochs **7, 8, and 9**.  
2. Restored **original SentencePiece tokenizers (Oct 2 models)** to fix the mismatch introduced by re-trained vocabularies.  
3. Added BOS/EOS consistency checks during inference.  
4. Automated detokenization and BLEU computation in the SLURM script.

---

### Scripts Overview

| File | Description |
|------|--------------|
| **`average_checkpoints_fixed.py`** | Python script that loads model weights from checkpoints 7, 8, and 9, averages their parameters (`θ_avg = (θ₇ + θ₈ + θ₉) / 3`), and saves the final averaged model. |
| **`average_checkpoints_fixed.sh`** | SLURM submission script that runs the averaging process on the Euler cluster, logs results, and saves averaged model checkpoints. |
| **`translate_avg_checkpoints_fixed.sh`** | Final translation and evaluation pipeline. Uses the fixed tokenizers, adds BOS/EOS tokens, performs inference, detokenizes output, and computes BLEU automatically. |
| **`resume_training.sh`** | Auxiliary script to resume training from the last saved checkpoint (not used in final experiment, kept for reproducibility). |

---

###  Results

| Model / Checkpoint | Val. BLEU | Test BLEU | Comments |
|--------------------|------------|------------|-----------|
| Assignment 1 Baseline | ~14.0 | **1.70** | Initial Transformer trained from scratch. |
| Checkpoint 7 | 25.85 | 1.30 | Highest validation BLEU; strong mid-training performance. |
| Checkpoint 8 | 23.75 | — | Slight decline; possible overfitting. |
| Checkpoint 9 | 21.97 | — | Stable training; final pre-averaging model. |
| Avg. (7+8+9) pre-fix | — | 0.36 | Tokenizer mismatch (retrained SentencePiece models). |
| Avg. (7+8+9) fixed | — | **0.24** | Correct tokenizers restored; BLEU remains low, likely due to destructive averaging. |

---

### Visualizations
- **`bleu_comparison.pdf`** — Validation vs. test BLEU across checkpoints.  
- **`tokenization_length.pdf`** — Shows effect of tokenization mismatch on output sequence length.  

---

### Discussion and Lessons Learned
- Even minor preprocessing drift (different SentencePiece models) can completely invalidate inference results.  
- Checkpoint averaging helps only when averaging **stable epochs**; including unstable early checkpoints (e.g., 7) degraded performance.  
- Strict tokenizer versioning and embedding consistency are crucial for reproducibility in MT pipelines.  
- Future work: try weighted averaging or later-epoch smoothing to reduce destructive interference across checkpoints.

---

**Repository structure:**

