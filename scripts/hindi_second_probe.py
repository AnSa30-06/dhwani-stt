"""On PURE Hindi, should the second decode be turbo or a Hindi specialist?

Since mix-first, an Indic clip decodes with the mix model, and if the output is
not code-switched a second decode runs and wins. Today that second decode is the
default model (large-v3-turbo). `scripts/model_compare.py` previously scored
vasista22/whisper-hindi-medium at 59.37/70 on these same five clips against the
shipped pair's 57.16 — and that finding was REJECTED at the time only because
routing pure-Hindi from code-switched needed an extra decode. Mix-first is that
router, so the rejection no longer applies and the number is worth re-testing on
the current scoring path.

Apache-2.0, so it clears the commercial-friendly bar (and more cleanly than the
OpenRAIL mix model).

    set HF_HOME=D:\\hf-cache
    python scripts/hindi_second_probe.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from mix_only_probe import load_clips, quality  # noqa: E402

HI_SPECIALIST = os.environ.get("HI_SPECIALIST", "vasista22/whisper-hindi-small")
HI_BEAM = int(os.environ.get("HI_BEAM", "1"))

_state = {}


def _hf_decode(D, pcm: bytes, repo: str, beam: int) -> str:
    """Transformers decode on an arbitrary whisper fine-tune, with the
    generation_config fallback this repo already needed once: vasista22's
    configs predate 2023 and reject task/language kwargs, which previously
    presented as a silent 0.00 across every clip."""
    import numpy as np
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    if repo not in _state:
        proc = AutoProcessor.from_pretrained(repo)
        model = WhisperForConditionalGeneration.from_pretrained(
            repo, low_cpu_mem_usage=True).eval()
        _state[repo] = (proc, model)
    proc, model = _state[repo]

    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    texts = []
    for start in range(0, len(audio), 30 * D.SR):
        piece = audio[start:start + 30 * D.SR]
        if len(piece) < int(0.2 * D.SR):
            continue
        feats = proc(piece, sampling_rate=D.SR, return_tensors="pt")["input_features"]
        feats = feats.to(dtype=model.dtype)
        kw = {"num_beams": beam} if beam > 1 else {}
        with torch.inference_mode():
            try:
                ids = model.generate(feats, task="transcribe", language="hi", **kw)
            except Exception:
                ids = model.generate(feats, **kw)      # pre-2023 generation_config
        texts.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
    text = " ".join(t for t in texts if t).strip()
    return D._normalize_numbers(D._deloop(D._text_from([(text, 0.0, 1.0)], "hi")))


def main():
    from solution import draft as D
    from mix_beam_probe import primary_only

    clips = [c for c in load_clips() if "hindi" in c["category"]]
    print(f"{len(clips)} pure-Hindi clips | specialist={HI_SPECIALIST}\n", flush=True)

    rows = []
    for c in clips:
        D.draft_reset()
        t = time.time(); p = primary_only(D, c["pcm"]); tp = time.time() - t
        t = time.time(); h = _hf_decode(D, c["pcm"], HI_SPECIALIST, HI_BEAM)
        th = time.time() - t
        qp, _, fp = quality(c, p)
        qh, _, fh = quality(c, h)
        rows.append({"clip_id": c["clip_id"], "primary": qp, "specialist": qh,
                     "delta": round(qh - qp, 2), "s_primary": round(tp, 1),
                     "s_specialist": round(th, 1), "flip_primary": fp,
                     "flip_specialist": fh, "pred_specialist": h})
        print(f"  {c['clip_id'][:30]:32s} turbo {qp:5.1f} ({tp:5.1f}s)   "
              f"specialist {qh:5.1f} ({th:5.1f}s)   {qh - qp:+6.1f}"
              f"{'  FLIP+' if (fh and not fp) else ''}", flush=True)

    n = len(rows) or 1
    mp = sum(r["primary"] for r in rows) / n
    mh = sum(r["specialist"] for r in rows) / n
    print(f"\nturbo (shipped second decode)  {mp:6.2f}/70   "
          f"{sum(r['s_primary'] for r in rows) / n:5.1f}s")
    print(f"{HI_SPECIALIST}  {mh:6.2f}/70   "
          f"{sum(r['s_specialist'] for r in rows) / n:5.1f}s")
    print(f"delta {mh - mp:+.2f}/70   new flips "
          f"{sum(1 for r in rows if r['flip_specialist'] and not r['flip_primary'])}/{n}")
    if not any(r["pred_specialist"].strip() for r in rows):
        raise SystemExit("every specialist output was blank -- nothing was measured "
                         "(this repo has hit a silent 0.00 from a stale "
                         "generation_config before)")
    json.dump(rows, open(Path(__file__).parent / "hindi_second_results.json", "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
