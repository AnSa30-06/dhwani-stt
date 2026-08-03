"""Is the PRIMARY decode earning its keep on Indic clips?

Today an Indic-detected clip is decoded TWICE: whisper first, then the router
escalates to the mix model, then _pick_mixed chooses. That second decode is the
expensive one (measured ~1.54x whisper) and it sits on the scored critical path,
which is where the end-to-final latency on Hindi/Hinglish clips comes from.

If the mix model alone is as good as the pair-and-pick, the primary decode can be
skipped whenever the language hint already says Indic -- halving the final's cost
on half the corpus for free. This measures exactly that, quality AND seconds:

    both  = the shipped path (_final_decode)
    mix   = the mix model alone, same orthography + number normalisation

Quality axis only; the seconds are this box's, but their RATIO transfers.

    set HF_HOME=D:\\hf-cache
    python scripts/mix_only_probe.py
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

INDIC_CATEGORIES = ("fleurs_hindi", "openslr104_hinglish")


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
            cat = row.get("category", "?")
            if not wav.exists() or row["clip_id"] in seen or cat not in INDIC_CATEGORIES:
                continue
            seen.add(row["clip_id"])
            with wave.open(str(wav), "rb") as w:
                pcm = w.readframes(w.getnframes())
            clips.append({"clip_id": row["clip_id"], "category": cat, "gold": row["gold"],
                          "gold_alternatives": row.get("gold_alternatives") or [],
                          "must_have": row.get("must_have") or [], "pcm": pcm})
    return clips


def quality(clip, pred):
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
    if not pred.strip():
        pts = 0.0
    elif err > 0.9:
        pts = min(pts, 14.0)
    if flipped:
        pts = min(pts, 35.0)
    return round(pts, 2), round(meaning, 3), flipped


def mix_only(D, pcm: str) -> str:
    """The router's escalation branch, on its own -- same helpers the shipped
    path runs it through, so the only difference is the missing primary."""
    if D._mix_backend() == "transformers":
        words, _ = D._transcribe_mix_transformers(pcm)
    else:
        words, _ = D._transcribe(pcm, "hi", prompt="", final=True, model=D._mix_model())
    return D._normalize_numbers(D._deloop(D._text_from(words, "hi")))


def main():
    from solution import draft as D

    clips = load_clips()
    print(f"{len(clips)} Indic clips | mix={D._mix_model()} "
          f"backend={D._mix_backend()} beam={D._beam_size(True)}\n", flush=True)

    rows = []
    for c in clips:
        D.draft_reset()
        t = time.time()
        both = D._final_decode(c["pcm"])
        t_both = time.time() - t

        D.draft_reset()
        t = time.time()
        only = mix_only(D, c["pcm"])
        t_only = time.time() - t

        qb, mb, fb = quality(c, both)
        qo, mo, fo = quality(c, only)
        rows.append({"clip_id": c["clip_id"], "category": c["category"],
                     "both": qb, "mix_only": qo, "delta": round(qo - qb, 2),
                     "flip_both": fb, "flip_mix": fo,
                     "s_both": round(t_both, 1), "s_mix": round(t_only, 1),
                     "pred_both": both, "pred_mix": only})
        print(f"  {c['clip_id'][:30]:32s} both {qb:5.1f} ({t_both:5.1f}s)   "
              f"mix-only {qo:5.1f} ({t_only:5.1f}s)   {qo - qb:+6.1f}"
              f"{'  FLIP+' if (fo and not fb) else ''}", flush=True)

    n = len(rows) or 1
    qb = sum(r["both"] for r in rows) / n
    qo = sum(r["mix_only"] for r in rows) / n
    sb = sum(r["s_both"] for r in rows) / n
    so = sum(r["s_mix"] for r in rows) / n
    print(f"\nboth      {qb:6.2f}/70   {sb:6.1f}s mean decode")
    print(f"mix-only  {qo:6.2f}/70   {so:6.1f}s mean decode")
    print(f"delta     {qo - qb:+6.2f}/70   {(1 - so / max(sb, 1e-9)) * 100:+5.1f}% decode time")
    print(f"new fact flips introduced by dropping the primary: "
          f"{sum(1 for r in rows if r['flip_mix'] and not r['flip_both'])}/{n}")
    if not any(r["pred_mix"].strip() for r in rows):
        raise SystemExit("every mix-only final was blank -- nothing was measured")
    json.dump(rows, open(Path(__file__).parent / "mix_only_results.json", "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
