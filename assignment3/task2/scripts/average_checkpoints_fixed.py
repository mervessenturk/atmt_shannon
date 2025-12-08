#!/usr/bin/env python3
import torch, os

# === Configuration ===
ckpt_dir = "/home/mesent/data/atmt_2025/cz-en/checkpoints"
files = [
    "checkpoint7_5.328.pt",
    "checkpoint8_6.310.pt",
    "checkpoint9_8.190.pt"
]
out_path = os.path.join(ckpt_dir, "checkpoint_averaged_fixed_args.pt")

print(f"\n🔄 Averaging {len(files)} checkpoints from: {ckpt_dir}")
for f in files:
    print("  -", f)

# === Averaging logic ===
avg_state, args_ref = None, None
for f in files:
    state = torch.load(os.path.join(ckpt_dir, f), map_location="cpu")
    if avg_state is None:
        avg_state = {k: v.clone() for k, v in state["model"].items()}
        args_ref = state.get("args", None)
    else:
        for k in avg_state:
            avg_state[k] += state["model"][k]

for k in avg_state:
    avg_state[k] /= len(files)

torch.save({"model": avg_state, "args": args_ref}, out_path)
print(f"\n✅ Saved averaged model with args to:\n{out_path}\n")
