#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-out/llama_interleaved_parity}"
BASELINE_OUT="$OUT_DIR/baseline.pt"
PIPER_DUMP_DIR="$OUT_DIR/piper"

rm -rf "$OUT_DIR"
mkdir -p "$PIPER_DUMP_DIR"

export PIPER_TEST_SEED="1337"
export PIPER_PARITY_DUMP_DIR="$PIPER_DUMP_DIR"

echo "[1/3] Running baseline (no piper) and logging tensors..."
python3 - <<'PY'
import os
import torch
from torch.nn.utils import parameters_to_vector

from test.models.llama import LLAMA_DEBUG, Transformer

seed = int(os.environ["PIPER_TEST_SEED"])
torch.manual_seed(seed)

batch_size = 16
seq_len = 256
mbs = 4
warmup = 1
iters = 1
steps = warmup + iters

config = LLAMA_DEBUG
loss_fn = torch.nn.CrossEntropyLoss()

x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
y = torch.randn((batch_size, seq_len, config.vocab_size))

model = Transformer(config, seq_len).cuda()
optim = torch.optim.Adam(model.parameters())

init_vector = parameters_to_vector([p.detach().cpu() for p in model.parameters()])
last_losses = []

for _ in range(steps):
    optim.zero_grad()
    step_losses = []
    for _ in range(mbs):
        logits = model(x.cuda())
        loss = loss_fn(logits, y.cuda())
        loss.backward()
        step_losses.append(loss.item())
    optim.step()
    last_losses = step_losses

final_vector = parameters_to_vector([p.detach().cpu() for p in model.parameters()])

torch.save(
    {
        "seed": seed,
        "init_vector": init_vector,
        "final_vector": final_vector,
        "last_step_losses": last_losses,
    },
    os.path.join(os.path.dirname(os.environ["PIPER_PARITY_DUMP_DIR"]), "baseline.pt"),
)
PY

echo "[2/3] Running piper end-to-end (interleaved 1f1b, mbs=4) ..."
python3 -m test.test_llama \
    --warmup 1 \
    --iters 1 \
    --schedule interleaved-1f1b \
    --mbs 4 \
    --model debug

echo "[3/3] Comparing logged outputs..."
python3 - <<'PY'
import glob
import os
import torch

out_dir = os.path.dirname(os.environ["PIPER_PARITY_DUMP_DIR"])
piper_dir = os.environ["PIPER_PARITY_DUMP_DIR"]
baseline = torch.load(os.path.join(out_dir, "baseline.pt"), map_location="cpu")

init_files = sorted(glob.glob(os.path.join(piper_dir, "rank*_stage*_init.pt")))
step_files = sorted(glob.glob(os.path.join(piper_dir, "rank*_step*.pt")))
assert init_files, "No piper init dumps found"
assert step_files, "No piper step dumps found"

stage_init = {}
for f in init_files:
    d = torch.load(f, map_location="cpu")
    stage_init[d["stage_id"]] = d["vector"]

latest_step = max(int(os.path.basename(f).split("step")[1].split(".")[0]) for f in step_files)
latest_files = [f for f in step_files if f"step{latest_step}.pt" in f]
stage_final = {}
all_losses = []
for f in latest_files:
    d = torch.load(f, map_location="cpu")
    stage_final.update(d["stage_to_vector"])
    all_losses.extend(d["losses"])

piper_init_vector = torch.cat([stage_init[s] for s in sorted(stage_init)])
piper_final_vector = torch.cat([stage_final[s] for s in sorted(stage_final)])

max_init_abs_diff = (baseline["init_vector"] - piper_init_vector).abs().max().item()
max_final_abs_diff = (baseline["final_vector"] - piper_final_vector).abs().max().item()

# Last 4 losses correspond to the timed iteration in this config (warmup=1, iters=1, mbs=4)
piper_last_step_losses = all_losses[-4:]
baseline_last_step_losses = baseline["last_step_losses"]
loss_diffs = [abs(a - b) for a, b in zip(baseline_last_step_losses, piper_last_step_losses)]
max_loss_abs_diff = max(loss_diffs)

print(f"max initial parameter abs diff: {max_init_abs_diff}")
print(f"max final parameter abs diff: {max_final_abs_diff}")
print(f"baseline last-step losses: {baseline_last_step_losses}")
print(f"piper last-step losses: {piper_last_step_losses}")
print(f"max last-step loss abs diff: {max_loss_abs_diff}")

assert len(baseline_last_step_losses) == len(piper_last_step_losses) == 4
assert max_init_abs_diff < 1e-8
assert max_final_abs_diff < 1e-5
assert max_loss_abs_diff < 1e-5
PY

echo "Done. Outputs available in: $OUT_DIR"
