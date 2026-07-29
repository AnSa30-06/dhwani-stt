"""Can decoding settings recover the English rare words that cost the facts axis?

The three failing FLEURS-English clips each lose their whole 20-point facts axis
(and get capped at 50) on a single mishearing: alkaline->acolyte, Sie->say,
Sintra->Cintra, splendours->splendorous. This probes whether beam search, the
temperature ladder, or previous-text conditioning recovers any of them on the
checkpoint we already have cached — no multi-GB download required.

    set HF_HOME=D:\\hf-cache
    python scripts/english_probe.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from scorecard import critical_flip, judge_meaning, wer  # noqa: E402

SR = 16000
MODEL = os.environ.get("DHWANI_MODEL", "large-v3-turbo")

VARIANTS = {
    "greedy (current)": dict(beam_size=1, temperature=[0.0]),
    "beam5": dict(beam_size=5, temperature=[0.0]),
    "beam5+ladder": dict(beam_size=5,
                         temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                         compression_ratio_threshold=2.4,
                         log_prob_threshold=-1.0, no_speech_threshold=0.6),
    "beam10": dict(beam_size=10, temperature=[0.0]),
    "beam5+context": dict(beam_size=5, temperature=[0.0],
                          condition_on_previous_text=True),
}


def load_targets():
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
        wav = ROOT / "data/dev/audio" / f"{cid}.wav"
        if not wav.exists():
            wav = ROOT / "samples" / f"{cid}.wav"
        if not wav.exists():
            continue
        with wave.open(str(wav), "rb") as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        out.append({**clip, "audio": pcm.astype(np.float32) / 32768.0})
    return out


def score(clip, pred):
    meaning = judge_meaning(clip["gold"], pred)
    flipped, _ = critical_flip(clip["gold"], pred, clip.get("must_have") or [])
    pts = 50 * meaning + (0 if flipped else 20)
    if flipped:
        pts = min(pts, 35.0)
    return round(pts, 2), flipped, round(meaning, 3)


def main():
    from faster_whisper import WhisperModel

    clips = load_targets()
    print(f"{len(clips)} English clips; model={MODEL}\n")
    t0 = time.time()
    model = WhisperModel(MODEL, device="cpu", compute_type="int8",
                         cpu_threads=os.cpu_count() or 4)
    print(f"loaded in {time.time() - t0:.0f}s\n")

    results = {}
    for label, kw in VARIANTS.items():
        total = 0.0
        flips = 0
        rows = []
        t1 = time.time()
        for clip in clips:
            params = dict(language="en", task="transcribe",
                          condition_on_previous_text=False, **kw)
            segs, _ = model.transcribe(clip["audio"], **params)
            pred = " ".join(s.text.strip() for s in segs).strip()
            pts, fl, mean = score(clip, pred)
            total += pts
            flips += fl
            missing = [t for t in (clip.get("must_have") or [])
                       if t.lower() not in pred.lower()]
            rows.append((clip["clip_id"][-4:], pts, fl, missing, pred))
        results[label] = (total / len(clips), flips, rows)
        print(f"=== {label}  mean {total/len(clips):.2f}/70  flips {flips}/{len(clips)}"
              f"  ({time.time()-t1:.0f}s)")
        for cid, pts, fl, missing, pred in rows:
            if fl:
                print(f"    {cid} {pts:5.1f} MISSING {missing}")
        print()

    print("=== summary ===")
    base = results["greedy (current)"][0]
    for label, (mean, flips, _) in results.items():
        print(f"  {label:18s} {mean:6.2f}/70  flips {flips}  ({mean-base:+.2f})")
    json.dump({k: {"mean": v[0], "flips": v[1],
                   "clips": [{"id": r[0], "pts": r[1], "flip": r[2],
                              "missing": r[3], "pred": r[4]} for r in v[2]]}
               for k, v in results.items()},
              open(Path(__file__).parent / "english_probe_results.json", "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
