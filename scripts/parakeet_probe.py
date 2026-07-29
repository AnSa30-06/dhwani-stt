"""NVIDIA Parakeet on the English clips — can it get the rare words Whisper misses?

Parakeet TDT 0.6B v2 tops the open English ASR leaderboards and is CC-BY-4.0,
so it is shippable here. English-only, which suits this engine: the router
already keeps Hindi and code-switched audio on the Whisper path, so Parakeet
would only ever handle the English clips whose losses are single rare-word
mishearings (alkaline, Sie, Sintra, splendours) that each forfeit the whole
20-point facts axis.

Run via onnx-asr (a few MB of runtime) rather than the full NeMo stack:

    pip install "onnx-asr[cpu,hub]"
    set HF_HOME=D:\\hf-cache
    python scripts/parakeet_probe.py
"""
from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scorecard import critical_flip, judge_meaning  # noqa: E402

SR = 16000
MODEL_ID = "nemo-parakeet-tdt-0.6b-v2"


def load_english():
    m = {}
    for f in ("data/dev/manifest.json", "samples/manifest.json"):
        p = ROOT / f
        if p.exists():
            for c in json.load(open(p, encoding="utf-8")):
                m.setdefault(c["clip_id"], c)
    out = []
    for cid, clip in m.items():
        if clip.get("category") != "fleurs_english":
            continue
        for d in ("data/dev/audio", "samples"):
            wav = ROOT / d / f"{cid}.wav"
            if wav.exists():
                with wave.open(str(wav), "rb") as w:
                    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                out.append({**clip, "audio": pcm.astype(np.float32) / 32768.0})
                break
    return out


def score(clip, pred):
    meaning = judge_meaning(clip["gold"], pred)
    flipped, _ = critical_flip(clip["gold"], pred, clip.get("must_have") or [])
    pts = 50 * meaning + (0 if flipped else 20)
    if flipped:
        pts = min(pts, 35.0)
    missing = [t for t in (clip.get("must_have") or []) if t.lower() not in pred.lower()]
    return round(pts, 2), flipped, round(meaning, 3), missing


def main():
    import onnx_asr

    clips = load_english()
    print(f"{len(clips)} English clips; model={MODEL_ID}", flush=True)
    t0 = time.time()
    model = onnx_asr.load_model(MODEL_ID)
    print(f"loaded in {time.time()-t0:.0f}s\n", flush=True)

    total = 0.0
    flips = 0
    rows = []
    for clip in clips:
        t = time.time()
        pred = model.recognize(clip["audio"], sample_rate=SR)
        dt = time.time() - t
        pts, fl, meaning, missing = score(clip, pred)
        total += pts
        flips += fl
        rows.append({"id": clip["clip_id"], "pts": pts, "flip": fl,
                     "meaning": meaning, "missing": missing, "pred": pred,
                     "secs": round(dt, 1)})
        print(f"  {clip['clip_id'][-4:]} {pts:5.1f}/70  meaning {meaning:.2f}"
              f"{'  FLIP ' + str(missing) if fl else ''}  ({dt:.1f}s)", flush=True)

    n = len(clips) or 1
    print(f"\nparakeet mean {total/n:.2f}/70   flips {flips}/{n}")
    print("(whisper turbo greedy on the same clips: 53.26/70, 3 flips;"
          " turbo beam5: 58.23/70, 2 flips)")
    json.dump({"mean": total / n, "flips": flips, "clips": rows},
              open(Path(__file__).parent / "parakeet_results.json", "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
