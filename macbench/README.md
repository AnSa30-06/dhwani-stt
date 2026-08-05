# Run this on the Mac

Thank you — this is the whole job:

```bash
cd dhwani-macbench
bash runall.sh
```

Then send back the `dhwani-results-<date>.zip` it prints at the end.

That is genuinely all. Everything else below is only if something looks wrong.

---

## What to expect

| | |
|---|---|
| **Time** | about 3 hours, mostly unattended. `QUICK=1 bash runall.sh` does the important half in about 45 minutes. |
| **Disk** | **~25 GB free**, almost all of it speech models landing in `~/.cache/huggingface`. Tight on space? `SKIP_BIG_MODELS=1 bash runall.sh` drops the four multi-gigabyte model-comparison rows and needs about 12 GB. Every latency result survives that. |
| **Memory** | 16 GB is fine, 24 GB is comfortable. Configurations run one at a time in separate processes, and the decode-rate probes deliberately load one model per process, so nothing ever stacks. |
| **Network** | needed at the start (model downloads), not during the measurement. |
| **Where it writes** | this folder, plus the huggingface model cache. Nothing else on your machine is touched and nothing is uploaded. |

The screen prints what it is doing. It is normal for it to sit quietly for
several minutes at a time — it is playing audio clips through a speech
recogniser in real time, so a clip that is 12 seconds long takes at least 12
seconds.

## If you need to stop it

Press `Ctrl-C`. Nothing is lost: re-running `bash runall.sh` picks up where it
stopped and skips everything already finished. Closing the lid is fine too.

## If something breaks

It is built to keep going. A step that fails writes down why it failed and the
run continues — a broken step is a useful result, not a reason to stop. So:

**Send the zip even if it printed errors.** That is more useful than a clean run
that never happened.

If it stops immediately, the likely cause is Python. It needs 3.10 or newer:

```bash
python3 --version
```

If that is missing or too old, `brew install python@3.12` and run it again.

## Options

```bash
SMOKE=1 bash runall.sh              # ~5 min: checks everything works before the long run
SKIP_BIG_MODELS=1 bash runall.sh    # ~12 GB instead of ~25 GB
QUICK=1 bash runall.sh              # ~45 min: the measurements most likely to matter
MACBENCH_MAX_HOURS=2 bash runall.sh # stop starting new work after 2 hours
SKIP_SETUP=1 bash runall.sh         # reuse the environment from an earlier run
```

## What it is actually doing, if you're curious

This is a speech-to-text engine for a competition. It was built on a Windows
laptop with no GPU, which decodes speech 10 to 20 times slower than the Mac it
gets scored on. Nearly a third of the score is *how fast the transcript comes
back*, and on that laptop that number could never be measured at all — every
decision about speed so far has been an educated guess.

Your Mac is the same kind of machine the scoring runs on. The run tries about
thirty different configurations of the engine, times each one properly, and
writes down which ones actually help.
