"""Does MIX-FIRST change quality, and what does it save?

Mix-first only engages when a language hint already says Indic (the streaming
partials or a pinned committed window supply it). So the two arms here are the
same _final_decode with and without that hint:

    no-hint  = today's path: primary decode, then escalate to the mix model
    hint=hi  = mix model first, and skip the primary when its output is clearly
               code-switched

Expected from the mix-only measurement: Hinglish identical quality at roughly
half the decode time, pure Hindi identical quality at the same cost (both models
still run, just in the other order).

    set HF_HOME=D:\\hf-cache
    python scripts/mix_first_probe.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mix_only_probe import load_clips, quality  # noqa: E402


def main():
    from solution import draft as D

    clips = load_clips()
    print(f"{len(clips)} Indic clips | mix={D._mix_model()} "
          f"latin_min={D.MIX_FIRST_LATIN_MIN}\n", flush=True)

    rows = []
    for c in clips:
        D.draft_reset()
        D._lang = None                      # no hint -> today's ordering
        t = time.time()
        base = D._final_decode(c["pcm"])
        t_base = time.time() - t

        D.draft_reset()
        D._lang = "hi"                      # the hint the partials supply
        t = time.time()
        first = D._final_decode(c["pcm"])
        t_first = time.time() - t
        D._lang = None

        qb, _, fb = quality(c, base)
        qf, _, ff = quality(c, first)
        rows.append({"clip_id": c["clip_id"], "category": c["category"],
                     "base": qb, "mix_first": qf, "delta": round(qf - qb, 2),
                     "s_base": round(t_base, 1), "s_first": round(t_first, 1),
                     "saved_pct": round((1 - t_first / max(t_base, 1e-9)) * 100, 1),
                     "flip_base": fb, "flip_first": ff,
                     "same_text": base.strip() == first.strip()})
        print(f"  {c['clip_id'][:30]:32s} base {qb:5.1f} ({t_base:5.1f}s)   "
              f"mix-first {qf:5.1f} ({t_first:5.1f}s)   {qf - qb:+6.1f}   "
              f"{rows[-1]['saved_pct']:+5.1f}% time"
              f"{'  FLIP+' if (ff and not fb) else ''}", flush=True)

    n = len(rows) or 1
    qb = sum(r["base"] for r in rows) / n
    qf = sum(r["mix_first"] for r in rows) / n
    sb = sum(r["s_base"] for r in rows) / n
    sf = sum(r["s_first"] for r in rows) / n
    print(f"\nbase       {qb:6.2f}/70   {sb:6.1f}s mean decode")
    print(f"mix-first  {qf:6.2f}/70   {sf:6.1f}s mean decode")
    print(f"delta      {qf - qb:+6.2f}/70   {(1 - sf / max(sb, 1e-9)) * 100:+5.1f}% decode time")
    print(f"new fact flips: {sum(1 for r in rows if r['flip_first'] and not r['flip_base'])}/{n}")
    for cat in sorted({r["category"] for r in rows}):
        sub = [r for r in rows if r["category"] == cat]
        print(f"   {cat:22s} delta {sum(r['delta'] for r in sub) / len(sub):+6.2f}/70   "
              f"time {sum(r['saved_pct'] for r in sub) / len(sub):+5.1f}%")
    json.dump(rows, open(Path(__file__).parent / "mix_first_results.json", "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
