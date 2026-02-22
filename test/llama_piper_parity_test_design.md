# LLaMA interleaved-1f1b parity workflow (shell-script approach)

## Goal
Compare baseline (no-piper) and piper end-to-end runs for the exact 4-microbatch interleaved 1f1b case.

## Why shell script
Per request, this uses a shell driver that runs the two cases separately, writes tensor/loss logs to output files, and compares the files afterward.

## Fixed case
- `model=debug`
- `schedule=interleaved-1f1b`
- `mbs=4`
- `warmup=1`
- `iters=1`

## Script behavior
`test/run_llama_interleaved_parity.sh`:
1. Runs a baseline eager path (no piper) and saves:
   - initial flattened parameter vector
   - final flattened parameter vector
   - losses from the last step (4 microbatches)
2. Runs piper exactly with:
   - `python3 -m test.test_llama --warmup 1 --iters 1 --schedule interleaved-1f1b --mbs 4 --model debug`
3. Reads piper dump files and compares against baseline:
   - max initial parameter abs diff
   - max final parameter abs diff
   - per-microbatch loss diffs for the last step

## Runtime logging support
- `src/piper_actor.py` adds dump hooks controlled by `PIPER_PARITY_DUMP_DIR`:
  - stage-init vectors at stage load
  - per-update vectors + losses at optimizer update
- `test/test_llama.py` reads optional `PIPER_TEST_SEED` to keep deterministic data generation.

## Output
By default all artifacts are written under:
- `out/llama_interleaved_parity/`
