"""Score the REAL streaming path: quality and end-to-final latency together.

Every other harness in this repo measures one half and is blind to the other:

  * shipped_config_score.py calls _final_decode directly, so it never runs
    _finalize -- it cannot see the budget being blown, and reports the quality
    of a transcript the shipped path may never actually return.
  * stream_ab.py sets DHWANI_SPECULATE=0 and drains the committers before
    finalizing, deliberately removing the timing behaviour.

This one reproduces evaluator.py: 20 ms frames paced at 1x real time, a partial
every DRAFT_EVERY_FRAMES frames, then `end` -- and it clocks end-to-final the
way streaming_scorecard does, from the last audio frame to the returned final.

Absolute milliseconds here are NOT the M1's. This box decodes several times
slower, so treat the ms as this machine's, and read `path` (which branch
_finalize took) and `rtf` (decode seconds per audio second) as the parts that
carry over.

    set HF_HOME=D:\\hf-cache
    python scripts/stream_latency.py --runs 1
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scorecard import critical_flip, has_repetition_loop, judge_meaning, phonetic_token_f1, wer  # noqa: E402
from streaming_scorecard import end_to_final_points  # noqa: E402

FRAME_S = 0.02
DRAFT_EVERY_FRAMES = 25          # matches solution/stream_server.py


def load_clips():
    clips, seen = [], set()
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
                          "gold": row["gold"],
                          "gold_alternatives": row.get("gold_alternatives") or [],
                          "must_have": row.get("must_have") or [], "pcm": pcm})
    return clips


def quality_points(clip, pred):
    """The 70-point quality axis, same shape as streaming_scorecard."""
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


def stream_once(D, pcm, speed=1.0):
    """Feed one clip exactly like the evaluator does and time the final."""
    D.draft_reset()
    D._LAST_FINAL_PATH = ""
    frame = int(FRAME_S * D.BYTES_PER_SEC)
    n_frames = max(1, len(pcm) // frame)
    dt = FRAME_S / speed

    t_start = time.monotonic()
    t_send = t_start
    since = 0
    for i in range(1, n_frames + 1):
        t_send += dt
        delay = t_send - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        since += 1
        if since >= DRAFT_EVERY_FRAMES:
            since = 0
            D.draft(pcm[: i * frame], False)

    # How much of the clip a background decode had already locked when `end`
    # arrived. This is the lever: whatever is covered here the final need not
    # decode, so `covered` close to 1.0 is what a fast final actually requires.
    with D._state_lock:
        covered = D._fc_bytes
    spec_covered = D._spec_covered

    t_end_audio = time.monotonic()
    text, _ = D.draft(pcm, True)
    end_to_final_ms = max(0.0, (time.monotonic() - t_end_audio) * 1000.0)
    return {"text": text, "ms": end_to_final_ms, "path": D._LAST_FINAL_PATH,
            "audio_s": len(pcm) / D.BYTES_PER_SEC,
            "covered": covered / max(1, len(pcm)),
            "spec_covered": spec_covered / max(1, len(pcm))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1, help="repeats per clip; median is scored")
    ap.add_argument("--speed", type=float, default=1.0, help="feed rate (1.0 = real time)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N clips")
    ap.add_argument("--category", default="", help="substring filter on clip category")
    ap.add_argument("--warm", action="store_true", help="load models before clip 1")
    ap.add_argument("--out", default="stream_latency_results.json")
    args = ap.parse_args()

    from solution import draft as D

    if args.warm:
        # The steady state every clip after the first one sees. Without this the
        # first clip pays the whole model load on its critical path, which is a
        # real defect of the sealed server (see the cold-start note in draft.py)
        # but drowns out everything else being measured here.
        t = time.monotonic()
        D.warm_models()
        print(f"warmed in {time.monotonic() - t:.0f}s", flush=True)

    clips = load_clips()
    if args.category:
        clips = [c for c in clips if args.category in c["category"]]
        if not clips:
            raise SystemExit(f"no clips matched category {args.category!r}")
    if args.limit:
        clips = clips[: args.limit]
    print(f"{len(clips)} clips | beam={D._beam_size(True)} mix={D._mix_model()} "
          f"chunk={D._chunk_s()}s budget={D._final_budget_s()}s "
          f"speculate={D._speculation_enabled()}\n", flush=True)

    rows = []
    for clip in clips:
        runs = [stream_once(D, clip["pcm"], args.speed) for _ in range(args.runs)]
        runs.sort(key=lambda r: r["ms"])
        median = runs[(len(runs) - 1) // 2]           # the median-latency run
        ms = statistics.median([r["ms"] for r in runs])
        pts, meaning, flip = quality_points(clip, median["text"])
        lat = end_to_final_points(ms)
        rows.append({"clip_id": clip["clip_id"], "category": clip["category"],
                     "quality": pts, "latency": round(lat, 2),
                     "total": round(pts + lat, 2), "ms": round(ms),
                     "path": median["path"], "meaning": meaning, "flip": flip,
                     "audio_s": round(median["audio_s"], 1), "pred": median["text"],
                     "covered": round(median["covered"], 2),
                     "spec_covered": round(median["spec_covered"], 2)})
        print(f"  {clip['clip_id'][:30]:32s} q{pts:5.1f} +lat{lat:5.1f} "
              f"= {pts + lat:5.1f}  {ms:6.0f}ms  {median['path'] or '-':12s}"
              f" cov{median['covered']:4.0%} spec{median['spec_covered']:4.0%}"
              f"{'  FLIP' if flip else ''}", flush=True)

    n = len(rows) or 1
    q = sum(r["quality"] for r in rows) / n
    latency = sum(r["latency"] for r in rows) / n
    print(f"\nQUALITY {q:.2f}/70   LATENCY {latency:.2f}/30   TOTAL {q + latency:.2f}/100")
    print(f"median end-to-final {statistics.median([r['ms'] for r in rows]):.0f}ms"
          f"   fact flips {sum(r['flip'] for r in rows)}/{n}")
    paths: dict[str, int] = {}
    for r in rows:
        paths[r["path"] or "-"] = paths.get(r["path"] or "-", 0) + 1
    print("final path taken: " + ", ".join(f"{k}={v}" for k, v in sorted(paths.items())))
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    for cat, vals in sorted(by_cat.items()):
        print(f"   {cat:24s} q{sum(v['quality'] for v in vals) / len(vals):6.2f} "
              f"lat{sum(v['latency'] for v in vals) / len(vals):6.2f}  ({len(vals)})")

    json.dump({"quality": q, "latency": latency, "total": q + latency, "clips": rows},
              open(Path(__file__).parent / args.out, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
