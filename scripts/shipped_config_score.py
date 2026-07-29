"""Score the SHIPPED configuration end to end, on every local clip.

model_compare.py measures one checkpoint at a time; this runs the real engine
path — _final_decode with the router, the mix model, beam search, the number
repair and the orthography adapter all live — so the number it prints is what
the shipped defaults actually produce, not an extrapolation.

Quality axis only. This laptop is not the M1, so latency here means nothing.

    set HF_HOME=D:\\hf-cache
    python scripts/shipped_config_score.py
"""
from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scorecard import critical_flip, judge_meaning, phonetic_token_f1, wer  # noqa: E402


def load_clips():
    m = {}
    for f in ("data/dev/manifest.json", "samples/manifest.json"):
        p = ROOT / f
        if p.exists():
            for c in json.load(open(p, encoding="utf-8")):
                m.setdefault(c["clip_id"], c)
    out = []
    for cid, clip in m.items():
        for d in ("data/dev/audio", "samples"):
            wav = ROOT / d / f"{cid}.wav"
            if wav.exists():
                with wave.open(str(wav), "rb") as w:
                    out.append({**clip, "pcm": w.readframes(w.getnframes())})
                break
    return out


def quality(clip, pred):
    refs = [clip["gold"], *(clip.get("gold_alternatives") or [])]
    best = None
    for i, ref in enumerate(refs):
        meaning = judge_meaning(ref, pred)
        if i > 0:
            meaning = max(meaning, phonetic_token_f1(ref, pred))
        err = wer(ref, pred)
        if best is None or meaning > best[1]:
            best = (ref, meaning, err)
    ref, meaning, err = best
    flipped, _ = critical_flip(ref, pred, clip.get("must_have") or [])
    pts = 50 * meaning + (0 if flipped else 20)
    if not pred.strip():
        pts = 0.0
    elif err > 0.9:
        pts = min(pts, 14.0)
    if flipped:
        pts = min(pts, 35.0)
    return round(pts, 2), flipped, round(meaning, 3)


def main():
    from solution import draft as D

    clips = load_clips()
    print(f"{len(clips)} clips | beam={D._beam_size(True)} "
          f"mix={D._mix_model()} chunk={D._chunk_s()}s\n", flush=True)

    total = 0.0
    flips = 0
    by_cat: dict[str, list] = {}
    rows = []
    for clip in clips:
        D.draft_reset()
        t = time.time()
        pred = D._final_decode(clip["pcm"])       # the real shipped path
        dt = time.time() - t
        pts, fl, meaning = quality(clip, pred)
        total += pts
        flips += fl
        by_cat.setdefault(clip.get("category", "?"), []).append(pts)
        rows.append({"id": clip["clip_id"], "cat": clip.get("category"),
                     "pts": pts, "flip": fl, "meaning": meaning, "pred": pred})
        print(f"  {clip['clip_id'][:30]:32s} {pts:5.1f}/70 m{meaning:.2f}"
              f"{'  FLIP' if fl else ''}  ({dt:.0f}s)", flush=True)

    n = len(clips) or 1
    print(f"\nSHIPPED CONFIG: {total/n:.2f}/70   fact flips {flips}/{n}")
    for cat, vals in sorted(by_cat.items()):
        print(f"   {cat:24s} {sum(vals)/len(vals):6.2f}/70  ({len(vals)} clips)")
    print("\nbaseline that scored 62.51 (turbo greedy, no router): 49.43/70, 6 flips")
    json.dump({"mean": total / n, "flips": flips, "clips": rows},
              open(Path(__file__).parent / "shipped_config_results.json", "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
