"""Can the mix model alone replace the PAIR on pure Hindi, if it gets a beam?

Two facts make this the question worth asking:

  * On pure Hindi `_pick_mixed` can never return the mix candidate — it demands
    >= _mix_latin_min() distinct Latin words and pure-Hindi output carries 0-3.
    So the second decode on those clips is discarded 100% of the time.
  * The mix model has always decoded GREEDILY, while the primary got beam 5
    (worth +4.97/70 on English). Since mix-first it produces the whole final on
    code-switched clips, so its beam width is now a scored parameter.

If beam closes the -5.8/70 gap that mix-only showed on pure Hindi, then EVERY
Indic clip collapses to a single decode with no quality loss.

    set HF_HOME=D:\\hf-cache
    python scripts/mix_beam_probe.py
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


def primary_only(D, pcm: bytes) -> str:
    """The default model on its own — no mix pass, no pick."""
    words, raw = D._transcribe(pcm, None, "", final=True)
    lang = "hi" if raw in D._INDIC else raw
    return D._normalize_numbers(D._deloop(D._text_from(words, lang)))


def mix_at_beam(D, pcm: bytes, beam: int) -> str:
    os.environ["DHWANI_MIX_BEAM"] = str(beam)
    words, _ = D._transcribe_mix_transformers(pcm)
    return D._normalize_numbers(D._deloop(D._text_from(words, "hi")))


def main():
    from solution import draft as D

    clips = load_clips()
    arms = [("primary", lambda pcm: primary_only(D, pcm)),
            ("mix_greedy", lambda pcm: mix_at_beam(D, pcm, 1)),
            ("mix_beam5", lambda pcm: mix_at_beam(D, pcm, 5))]
    print(f"{len(clips)} Indic clips | mix={D._mix_model()}\n", flush=True)

    rows = []
    for c in clips:
        row = {"clip_id": c["clip_id"], "category": c["category"]}
        for name, fn in arms:
            D.draft_reset()
            t = time.time()
            pred = fn(c["pcm"])
            dt = time.time() - t
            pts, _, flip = quality(c, pred)
            row[name] = pts
            row[f"{name}_s"] = round(dt, 1)
            row[f"{name}_flip"] = flip
            row[f"{name}_pred"] = pred
        rows.append(row)
        print(f"  {c['clip_id'][:30]:32s} "
              + "  ".join(f"{n} {row[n]:5.1f} ({row[n + '_s']:5.1f}s)" for n, _ in arms),
              flush=True)

    def mean(rs, k):
        return sum(r[k] for r in rs) / max(1, len(rs))

    for cat in ["ALL"] + sorted({r["category"] for r in rows}):
        sub = rows if cat == "ALL" else [r for r in rows if r["category"] == cat]
        print(f"\n=== {cat} ({len(sub)} clips) ===")
        for name, _ in arms:
            print(f"   {name:12s} {mean(sub, name):6.2f}/70   "
                  f"{mean(sub, name + '_s'):6.1f}s   "
                  f"flips {sum(r[name + '_flip'] for r in sub)}/{len(sub)}")
        print(f"   beam gain on the mix model: "
              f"{mean(sub, 'mix_beam5') - mean(sub, 'mix_greedy'):+6.2f}/70")
        print(f"   mix_beam5 vs primary:       "
              f"{mean(sub, 'mix_beam5') - mean(sub, 'primary'):+6.2f}/70")

    if not any(r["mix_beam5_pred"].strip() for r in rows):
        raise SystemExit("every mix_beam5 output was blank -- nothing was measured")
    json.dump(rows, open(Path(__file__).parent / "mix_beam_results.json", "w",
                         encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
