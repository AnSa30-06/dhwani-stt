"""Compare candidate FINAL models on the local validation set — quality axis only.

Latency on this CPU box is meaningless (the scoring box is an M1 Pro with the
accelerated backend) and is deliberately excluded; what transfers is the
transcript each model produces for a clip, scored with builderr's REAL
streaming scorer (meaning 50 + facts 20, with the quality caps applied).

Validation set = every clip with audio on disk from:
    data/dev/manifest.json   (audio in data/dev/audio/,  fetched by fetch_audio.py)
    samples/manifest.json    (audio in samples/)

Models (pick with --models, default all):
    turbo    faster-whisper large-v3-turbo int8 — the engine's shipped default
    apex     Oriserve/Whisper-Hindi2Hinglish-Apex   (transformers, Apache-2.0)
    zerostt  shunyalabs/zero-stt-hinglish           (transformers, OpenRAIL)

Each model decodes the way solution/draft.py's _final_decode does: an auto pass,
then a forced-Hindi retry when the auto pass looks wrong, _pick_better keeping
the cleaner candidate, then the engine's own de-loop + number normalization.

Run (Windows):
    set HF_HOME=D:\\hf-cache
    python scripts/model_compare.py            # all models, all clips
    python scripts/model_compare.py --models turbo apex --limit 4

Writes scripts/model_compare_results.json and prints a per-category table.
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
from solution import draft as D  # noqa: E402  (helpers only; no engine state used)

SR = 16000


# --- audio ------------------------------------------------------------------

def read_wav_16k_mono(path: Path):
    import numpy as np

    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR, f"{path}: {w.getframerate()}Hz, expected 16k"
        assert w.getnchannels() == 1, f"{path}: {w.getnchannels()}ch, expected mono"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


def load_clips() -> list[dict]:
    clips = []
    for manifest, audio_dir in (
        (ROOT / "data/dev/manifest.json", ROOT / "data/dev/audio"),
        (ROOT / "samples/manifest.json", ROOT / "samples"),
    ):
        if not manifest.exists():
            continue
        for row in json.load(open(manifest, encoding="utf-8")):
            wav = audio_dir / f"{row['clip_id']}.wav"
            if not wav.exists():
                continue
            clips.append({
                "clip_id": row["clip_id"],
                "category": row.get("category", "?"),
                "gold": row["gold"],
                "gold_alternatives": row.get("gold_alternatives") or [],
                "must_have": row.get("must_have") or [],
                "wav": str(wav),
            })
    # a clip id can appear in both manifests; keep the first (dev gold wins)
    seen, out = set(), []
    for c in clips:
        if c["clip_id"] not in seen:
            seen.add(c["clip_id"])
            out.append(c)
    return out


# --- decoding ---------------------------------------------------------------

class TurboCT2:
    name = "turbo"
    model_id = "large-v3-turbo"
    license = "MIT (OpenAI Whisper weights, Systran CT2 conversion)"

    def __init__(self):
        from faster_whisper import WhisperModel
        self.m = WhisperModel(self.model_id, device="cpu", compute_type="int8",
                              cpu_threads=os.cpu_count() or 4)

    def decode(self, audio, lang):
        segs, info = self.m.transcribe(
            audio, language=lang, task="transcribe", beam_size=1,
            condition_on_previous_text=False,
            temperature=list(D._FINAL_TEMPS),
            compression_ratio_threshold=2.4, log_prob_threshold=-1.0,
            no_speech_threshold=0.6)
        return " ".join(s.text.strip() for s in segs).strip(), getattr(info, "language", None)


class HFWhisper:
    def __init__(self, name, model_id, license_str):
        self.name, self.model_id, self.license = name, model_id, license_str
        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model_id)
        try:
            dtype = torch.bfloat16 if torch.cpu.is_bf16_supported() else torch.float32
        except AttributeError:
            dtype = torch.float32   # this torch build has no CPU bf16 probe
        self.m = WhisperForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=dtype, low_cpu_mem_usage=True)
        self.m.eval()

    def decode(self, audio, lang):
        t = self.torch
        inputs = self.proc(audio, sampling_rate=SR, return_tensors="pt")
        feats = inputs["input_features"].to(self.m.dtype)
        kw = {"task": "transcribe"}
        if lang:
            kw["language"] = lang
        with t.inference_mode():
            ids = self.m.generate(feats, **kw)
        text = self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
        return text, lang  # transformers generate doesn't surface detected lang simply


def engine_style_final(decoder, audio) -> str:
    """The _final_decode recipe: auto pass, forced-hi retry if it looks wrong,
    _pick_better, de-loop, number normalization."""
    try:
        text, raw = decoder.decode(audio, None)
        text = D._deloop(text)
    except Exception as e:  # noqa: BLE001
        print(f"    [auto pass failed: {type(e).__name__}: {e}]")
        text, raw = "", None
    if (not text or D._looks_bad(text)
            or (raw in D._INDIC and raw != "hi")
            or (raw not in ("en", "hi") and raw is not None)):
        try:
            cand = D._deloop(decoder.decode(audio, "hi")[0])
            text = D._pick_better(text, cand) if text else cand
        except Exception as e:  # noqa: BLE001
            print(f"    [hi retry failed: {type(e).__name__}: {e}]")
    return D._normalize_numbers(text or "")


# --- scoring (quality axis of the real streaming scorer) --------------------

def quality_points(clip, pred) -> dict:
    refs = [clip["gold"], *clip["gold_alternatives"]]
    best = None
    for i, ref in enumerate(refs):
        meaning = judge_meaning(ref, pred)
        err = wer(ref, pred)
        if i > 0:
            meaning = max(meaning, phonetic_token_f1(ref, pred))
        if best is None or meaning > best[1]:
            best = (ref, meaning, err)
    ref, meaning, err = best
    flipped, reasons = critical_flip(ref, pred, clip["must_have"])
    pts = 50.0 * meaning + (0.0 if flipped else 20.0)
    cap = None
    if not pred.strip():
        cap, reasons = 0.0, reasons + ["blank final"]
    elif has_repetition_loop(pred):
        cap, reasons = 30.0, reasons + ["repetition loop"]
    elif err > 0.9:
        cap, reasons = 20.0, reasons + [f"unrelated (WER {err:.2f})"]
    if flipped and (cap is None or cap > 50.0):
        cap = 50.0
    # quality axis is 70 of the cap's 100; scale caps into the 70-point frame
    if cap is not None:
        pts = min(pts, cap * 0.7)
    return {"points": round(pts, 2), "meaning": round(meaning, 3),
            "wer": round(err, 3), "flipped": flipped, "reasons": reasons}


class CT2Model(TurboCT2):
    """Any faster-whisper checkpoint, same final-quality decode settings."""

    def __init__(self, name, model_id, license_str):
        self.name, self.model_id, self.license = name, model_id, license_str
        from faster_whisper import WhisperModel
        self.m = WhisperModel(model_id, device="cpu", compute_type="int8",
                              cpu_threads=os.cpu_count() or 4)


MODELS = {
    "turbo": lambda: TurboCT2(),
    # The English fact flips (Sie->say, Sintra->Cintra, alkaline->acolyte) are
    # rare-word errors; a full large model is the obvious candidate to fix them.
    "largev3": lambda: CT2Model("largev3", "large-v3", "MIT (OpenAI weights)"),
    "medium": lambda: CT2Model("medium", "medium", "MIT (OpenAI weights)"),
    "apex": lambda: HFWhisper("apex", "Oriserve/Whisper-Hindi2Hinglish-Apex",
                              "Apache-2.0"),
    # Hindi is the weakest category (57.18/70) and is limited by meaning, not
    # fact flips -- both current models sit near 51 raw on pure Hindi, so
    # neither is actually good at it. A Hindi-specialist fine-tune is the
    # untested lever.
    "hindilarge": lambda: HFWhisper("hindilarge", "vasista22/whisper-hindi-large-v2",
                                    "MIT (check card)"),
    "zerostt": lambda: HFWhisper("zerostt", "shunyalabs/zero-stt-hinglish",
                                 "OpenRAIL"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--limit", type=int, default=0, help="clips per category, 0=all")
    ap.add_argument("--category", default="", help="only clips whose category contains this")
    ap.add_argument("--out", default="model_compare_results.json")
    args = ap.parse_args()

    clips = load_clips()
    if args.category:
        clips = [c for c in clips if args.category in c["category"]]
    if args.limit:
        by_cat: dict[str, list] = {}
        for c in clips:
            by_cat.setdefault(c["category"], []).append(c)
        clips = [c for cat in by_cat.values() for c in cat[: args.limit]]
    print(f"{len(clips)} clips with audio:",
          {c: sum(1 for x in clips if x['category'] == c) for c in {x['category'] for x in clips}})

    results = {}
    for mname in args.models:
        print(f"\n=== {mname} ===")
        t0 = time.time()
        try:
            dec = MODELS[mname]()
        except Exception as e:  # noqa: BLE001
            print(f"  LOAD FAILED: {type(e).__name__}: {e}")
            results[mname] = {"load_error": f"{type(e).__name__}: {e}"}
            continue
        print(f"  loaded in {time.time() - t0:.0f}s")
        rows = []
        for clip in clips:
            audio = read_wav_16k_mono(Path(clip["wav"]))
            t1 = time.time()
            pred = engine_style_final(dec, audio)
            q = quality_points(clip, pred)
            rows.append({**q, "clip_id": clip["clip_id"], "category": clip["category"],
                         "pred": pred, "decode_s": round(time.time() - t1, 1)})
            print(f"  {clip['clip_id'][:36]:36s} {q['points']:5.1f}/70"
                  f"  meaning {q['meaning']:.2f}  {'FLIP ' if q['flipped'] else ''}"
                  f"({rows[-1]['decode_s']}s)")
        results[mname] = {"clips": rows}
        del dec

    print("\n=== per-category quality (points/70, mean) ===")
    cats = sorted({c["category"] for c in clips})
    header = "model    " + "".join(f"{c[:18]:>20s}" for c in cats) + f"{'OVERALL':>10s}"
    print(header)
    for mname, res in results.items():
        if "clips" not in res:
            print(f"{mname:8s}  LOAD FAILED")
            continue
        cells = []
        for cat in cats:
            vals = [r["points"] for r in res["clips"] if r["category"] == cat]
            cells.append(f"{sum(vals) / len(vals):20.2f}" if vals else f"{'—':>20s}")
        overall = sum(r["points"] for r in res["clips"]) / len(res["clips"])
        print(f"{mname:8s}" + "".join(cells) + f"{overall:10.2f}")

    out = Path(__file__).parent / args.out
    json.dump(results, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
