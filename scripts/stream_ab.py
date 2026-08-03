"""Real-audio A/B for the chunked final: stream each validation clip through the
actual draft() path and score the final, once with the whole-buffer decode
(DHWANI_CHUNK_S=0) and once with chunking forced on (a small window, so the
10-15s local clips actually split — the shipped 24s window never triggers on
clips this short). If the two score columns match, the chunk/commit/join
machinery preserves quality on real audio; any drop is a boundary artifact.

Quality axis only (this CPU box is not the M1; latency here is meaningless).

    set HF_HOME=D:\\hf-cache
    python scripts/stream_ab.py --window 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scorecard import critical_flip, has_repetition_loop, judge_meaning, phonetic_token_f1, wer  # noqa: E402


def load_clips():
    clips = []
    seen = set()
    for manifest, audio_dir in (
        (ROOT / "data/dev/manifest.json", ROOT / "data/dev/audio"),
        (ROOT / "samples/manifest.json", ROOT / "samples"),
    ):
        if not manifest.exists():
            continue
        for row in json.load(open(manifest, encoding="utf-8")):
            wav = audio_dir / f"{row['clip_id']}.wav"
            if not wav.exists() or row["clip_id"] in seen:
                continue
            seen.add(row["clip_id"])
            with wave.open(str(wav), "rb") as w:
                pcm = w.readframes(w.getnframes())
            clips.append({"clip_id": row["clip_id"], "category": row.get("category", "?"),
                          "gold": row["gold"], "gold_alternatives": row.get("gold_alternatives") or [],
                          "must_have": row.get("must_have") or [], "pcm": pcm})
    return clips


def quality_points(clip, pred):
    refs = [clip["gold"], *clip["gold_alternatives"]]
    best = None
    for i, ref in enumerate(refs):
        meaning = judge_meaning(ref, pred)
        if i > 0:
            meaning = max(meaning, phonetic_token_f1(ref, pred))
        err = wer(ref, pred)
        if best is None or meaning > best[1]:
            best = (ref, meaning, err)
    ref, meaning, err = best
    flipped, _ = critical_flip(ref, pred, clip["must_have"])
    pts = 50.0 * meaning + (0.0 if flipped else 20.0)
    cap = None
    if not pred.strip():
        cap = 0.0
    elif has_repetition_loop(pred):
        cap = 30.0
    elif err > 0.9:
        cap = 20.0
    if flipped and (cap is None or cap > 50.0):
        cap = 50.0
    if cap is not None:
        pts = min(pts, cap * 0.7)
    return round(pts, 2), round(meaning, 3), flipped


def stream_final(D, pcm, step_s=1.0):
    """Feed the clip through draft() like the server does, letting the background
    committer close windows, then take the final. Not real-time (offline quality
    run), so we drain committers before the final instead of pacing at 1x."""
    D.draft_reset()
    step = int(step_s * D.BYTES_PER_SEC)
    for end in range(step, len(pcm) + 1, step):
        D.draft(pcm[:end], False)
    # Let any in-flight / pending window close before finalizing. This wait must
    # outlast a REAL window decode: at 30s it expired mid-commit on this CPU box,
    # the committer locked nothing, and the chunked arm quietly ran the control's
    # whole-buffer path -- which is why an earlier run printed identical scores
    # on both arms. Generous by design; this is an offline quality run.
    deadline = time.monotonic() + float(os.environ.get("DHWANI_AB_DRAIN_S", "300"))
    last, stable = -1, 0
    while time.monotonic() < deadline:
        D.draft(pcm, False)
        with D._state_lock:
            fb, busy = D._fc_bytes, D._fc_busy
        if not busy and fb == last:
            stable += 1
            if stable >= 3:
                break
        else:
            stable, last = 0, fb
        time.sleep(0.05)
    with D._state_lock:
        committed = D._fc_bytes
    text, _ = D.draft(pcm, True)
    # How much of the clip the committer had locked before the final ran. If
    # this is 0 on the chunked arm then chunking never engaged and an identical
    # score on both arms means "the same code path ran twice", not "chunking is
    # free". Both arms printing the same number is otherwise indistinguishable.
    return text, committed / max(1, len(pcm))


def run(clips, window):
    os.environ["DHWANI_MIX_MODEL"] = ""      # isolate chunking: turbo on both sides
    os.environ["DHWANI_SPECULATE"] = "0"
    # This is a QUALITY A/B, so the final must be allowed to finish. Without
    # this the shipped 3s budget expires long before a real decode returns on a
    # CPU box, every final falls through blank, and the run prints a clean
    # 0.00-vs-0.00 "no difference" -- which reads as "chunking is neutral" and
    # is actually "nothing was measured". That is how this A/B silently
    # produced nothing the first time it was attempted.
    os.environ.setdefault("DHWANI_FINAL_BUDGET_S", "600")
    import importlib
    import solution.draft as D
    importlib.reload(D)

    results = {}
    for label, chunk in (("whole", "0"), (f"chunk{window}", str(window))):
        os.environ["DHWANI_CHUNK_S"] = chunk
        print(f"\n=== {label} (DHWANI_CHUNK_S={chunk}) ===", flush=True)
        rows = []
        for c in clips:
            pred, covered = stream_final(D, c["pcm"])
            pts, meaning, flip = quality_points(c, pred)
            rows.append({"clip_id": c["clip_id"], "category": c["category"],
                         "points": pts, "meaning": meaning, "flip": flip,
                         "pred": pred, "covered": round(covered, 2)})
            print(f"  {c['clip_id'][:34]:34s} {pts:5.1f}/70  m{meaning:.2f}"
                  f"  committed {covered:4.0%}{'  FLIP' if flip else ''}", flush=True)
        if chunk != "0" and not any(r["covered"] for r in rows):
            raise SystemExit(
                f"[{label}] the committer never locked a single window, so this "
                f"arm ran the SAME whole-buffer path as the control. An equal "
                f"score here would mean nothing was compared.")
        if not any(r["pred"].strip() for r in rows):
            raise SystemExit(
                f"[{label}] every final came back blank -- the decoder never "
                f"produced anything, so this run measured NOTHING. Do not read "
                f"the 0.00 as a result. Check DHWANI_FINAL_BUDGET_S and HF_HOME.")
        results[label] = rows
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=8, help="forced chunk seconds for the B run")
    ap.add_argument("--per-category", type=int, default=0,
                    help="cap clips per category; this box needs ~1 min per clip per side")
    args = ap.parse_args()
    clips = load_clips()
    if args.per_category:
        seen: dict[str, int] = {}
        kept = []
        for c in clips:
            n = seen.get(c["category"], 0)
            if n < args.per_category:
                seen[c["category"]] = n + 1
                kept.append(c)
        clips = kept
    print(f"{len(clips)} clips; forcing chunk window = {args.window}s so short clips split")
    results = run(clips, args.window)

    labels = list(results)
    print("\n=== per-clip quality (points/70) ===")
    print(f"{'clip':36s}{labels[0]:>10s}{labels[1]:>12s}{'delta':>8s}")
    deltas = []
    a = {r["clip_id"]: r for r in results[labels[0]]}
    for r in results[labels[1]]:
        base = a[r["clip_id"]]["points"]
        d = round(r["points"] - base, 2)
        deltas.append(d)
        print(f"{r['clip_id'][:34]:34s}{base:10.1f}{r['points']:12.1f}{d:+8.1f}")
    n = len(deltas) or 1
    print(f"\nmean {labels[0]} {sum(a[r['clip_id']]['points'] for r in results[labels[1]])/n:.2f}"
          f"   mean {labels[1]} {sum(r['points'] for r in results[labels[1]])/n:.2f}"
          f"   mean delta {sum(deltas)/n:+.2f}")
    out = Path(__file__).parent / "stream_ab_results.json"
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
