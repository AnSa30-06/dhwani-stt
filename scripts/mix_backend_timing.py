"""How much of the final's wall-clock is the mix decode?

The Hindi path runs two decodes: the primary (CTranslate2 / mlx) and the mix
model via transformers, which is the slowest backend in the stack. If the mix
decode dominates, it is the thing to attack for latency — and converting the
mix model to CTranslate2 would cut it without giving up the Latin-script output
that the facts axis depends on.

Absolute times here describe this laptop, not the M1 scoring box; the RATIO is
what transfers.

    set HF_HOME=D:\\hf-cache
    python scripts/mix_backend_timing.py
"""
from __future__ import annotations

import os
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLIP = ROOT / "samples/openslr104_hi_en_103085_w5Jyq3XMbb3WwiKQ_0001.wav"


def main():
    os.environ.setdefault("DHWANI_MIX_MODEL", "shunyalabs/zero-stt-hinglish")
    from solution import draft as D

    with wave.open(str(CLIP), "rb") as w:
        pcm = w.readframes(w.getnframes())
    dur = len(pcm) / D.BYTES_PER_SEC
    print(f"clip {dur:.1f}s\n")

    # primary decode (CTranslate2 here, mlx on the scoring box)
    t = time.time()
    D._transcribe(pcm, "hi", "", final=True, fast=True)
    primary_warm = time.time() - t
    t = time.time()
    D._transcribe(pcm, "hi", "", final=True, fast=True)
    primary = time.time() - t
    print(f"primary  (ctranslate2, int8) : first {primary_warm:6.2f}s   warm {primary:6.2f}s")

    # mix decode via transformers
    t = time.time()
    D._transcribe_mix_transformers(pcm)
    mix_warm = time.time() - t
    t = time.time()
    D._transcribe_mix_transformers(pcm)
    mix = time.time() - t
    print(f"mix      (transformers)      : first {mix_warm:6.2f}s   warm {mix:6.2f}s")

    print()
    print(f"mix / primary ratio (warm): {mix / max(primary, 1e-6):.1f}x")
    print(f"a Hindi final pays both   : {primary + mix:.2f}s of decode on this box")
    if mix > primary * 1.5:
        print("\n=> the mix decode dominates; converting it to CTranslate2 is the")
        print("   latency lever, and it keeps the Latin output the facts axis needs.")
    else:
        print("\n=> the mix decode is not the bottleneck; look elsewhere for latency.")


if __name__ == "__main__":
    main()
