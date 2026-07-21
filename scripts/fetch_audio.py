"""Fetch each clip's WAV by provenance into data/<split>/audio/<clip_id>.wav.

Runs once, online, BEFORE scoring (scoring itself is network-blocked). FLEURS clips
come straight from HuggingFace by id — reproducible anywhere. OpenSLR/YouTube clips
are staged from a directory you point at (your private audio store on the Action via
a secret), matched by source_audio filename.

  python scripts/fetch_audio.py --manifest data/dev/manifest.json --out data/dev/audio
  RAMBLEFIX_AUDIO_DIR=/path/to/staged python scripts/fetch_audio.py --manifest data/hidden/manifest.json --out data/hidden/audio
"""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path

_FLEURS_CACHE = {}


def _fleurs_row(config: str, split: str, fleurs_id: int):
    key = (config, split)
    if key not in _FLEURS_CACHE:
        from datasets import Audio, load_dataset
        ds = load_dataset("google/fleurs", config, split=split)
        ds = ds.cast_column("audio", Audio(decode=False))  # raw bytes; no torchcodec
        _FLEURS_CACHE[key] = {int(r["id"]): r for r in ds}
    return _FLEURS_CACHE[key].get(fleurs_id)


def fetch_one(clip: dict, out_dir: Path, staged: Path | None) -> str:
    ref = clip.get("audio_ref", {})
    dest = out_dir / f"{clip['clip_id']}.wav"
    if dest.exists():
        return "cached"
    # 1) FLEURS via HuggingFace (fully reproducible by id). datasets>=5 needs
    #    torchcodec to DECODE audio; reading the raw bytes (decode=False) and
    #    converting with soundfile avoids that — FLEURS ships float32 WAVs,
    #    which also defeats the stdlib wave module, so go through soundfile.
    if ref.get("repo") == "google/fleurs":
        try:
            import io
            import numpy as np
            import soundfile as sf
            fid = int(str(ref["id"]).rsplit("_", 1)[-1])  # fleurs_en_us_test_1904 -> 1904
            row = _fleurs_row(ref["config"], ref["split"], fid)
            if not row:
                return "missing(fleurs id not found)"
            data, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16)
            sf.write(str(dest), pcm, sr, subtype="PCM_16")
            return "fleurs"
        except Exception as e:  # noqa: BLE001
            return f"error(fleurs:{type(e).__name__})"
    # 2) staged audio store (OpenSLR / YouTube / recorded), matched by filename
    src_name = ref.get("source_audio") or ""
    if staged and src_name:
        cand = staged / src_name
        if cand.exists():
            shutil.copyfile(cand, dest)
            return "staged"
    return "missing(no source)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    manifest = json.load(open(args.manifest))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    staged = os.environ.get("RAMBLEFIX_AUDIO_DIR")
    staged = Path(staged) if staged else None

    from collections import Counter
    tally = Counter()
    for clip in manifest:
        tally[fetch_one(clip, out_dir, staged)] += 1
    print(f"fetched into {out_dir}:")
    for k, v in sorted(tally.items()):
        print(f"  {k:24s} {v}")
    missing = sum(v for k, v in tally.items() if k.startswith("missing") or k.startswith("error"))
    if missing:
        print(f"NOTE: {missing} clips not fetched — FLEURS needs network; OpenSLR/YouTube need RAMBLEFIX_AUDIO_DIR staged.")


if __name__ == "__main__":
    main()
