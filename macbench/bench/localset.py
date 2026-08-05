"""Merge every clip that actually has audio into one manifest the evaluator can read.

The corpus is split across two manifests (data/dev holds the FLEURS rows,
samples/ holds the OpenSLR Hinglish rows) and the official manifest lists rows
whose audio is not redistributable. evaluator.py resolves audio as
<manifest dir>/audio/<clip_id>.wav and raises on the first row it cannot find,
so it needs a single directory containing exactly the rows we can play.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "local"

SOURCES = [
    (ROOT / "data/dev/manifest.json", ROOT / "data/dev/audio"),
    (ROOT / "samples/manifest.json", ROOT / "samples"),
]


def build(verbose: bool = True) -> Path:
    audio_dir = OUT_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows, seen, missing = [], set(), []

    for manifest, src_dir in SOURCES:
        if not manifest.exists():
            continue
        for row in json.load(open(manifest, encoding="utf-8")):
            cid = row["clip_id"]
            if cid in seen:
                continue
            wav = src_dir / f"{cid}.wav"
            if not wav.exists():
                missing.append(cid)
                continue
            seen.add(cid)
            dest = audio_dir / f"{cid}.wav"
            if not dest.exists():
                shutil.copyfile(wav, dest)
            rows.append(row)

    long_row = _build_long_english(rows, audio_dir)
    if long_row:
        rows.append(long_row)

    # Keep categories interleaved. evaluator.py's --per-category walks the list
    # in order, and a manifest grouped by category would hand a truncated run
    # the same category twice before it ever reached the third.
    rows.sort(key=lambda r: (_seq(rows, r), str(r.get("category"))))

    # ensure_ascii deliberately: evaluator.py opens the manifest with
    # `json.load(open(path))`, i.e. the platform's default codec. That is UTF-8
    # on the scoring Mac and cp1252 on Windows, where the Devanagari gold makes
    # it raise before the run starts. Escaped JSON is the identical document and
    # loads under any codec, so the harness is testable on both.
    out = OUT_DIR / "manifest.json"
    out.write_text(json.dumps(rows, ensure_ascii=True, indent=1) + "\n", encoding="utf-8")
    if verbose:
        cats: dict[str, int] = {}
        for r in rows:
            cats[str(r.get("category"))] = cats.get(str(r.get("category")), 0) + 1
        print(f"local corpus: {len(rows)} clips with audio -> {out}")
        for k, v in sorted(cats.items()):
            print(f"   {k:26s} {v}")
        if missing:
            print(f"   ({len(missing)} manifest rows have no local audio and were skipped)")
    return out


def _build_long_english(rows: list, audio_dir: Path) -> dict | None:
    """Stitch the English clips into one ~60s clip, because the hidden corpus
    has a category we cannot otherwise test at all.

    docs/STREAMING_CONTRACT.md: the official manifest is 96 rows covering
    "English, Hindi, mixed Hindi-English, and longer English speech". Every clip
    we hold is 6-15 seconds. Long audio is a different code path — it is the
    only case where the window committer actually closes anything, where the
    final faces a bounded tail rather than the whole buffer, and where whisper
    needs more than one 30-second encoder window.

    The meaning score on a stitched gold is approximate and should be read as
    such. The LATENCY measurement is not approximate at all, and latency on long
    clips is exactly the thing no harness here has ever been able to look at.
    """
    import wave

    english = [r for r in rows if str(r.get("category")) == "fleurs_english"]
    if len(english) < 4:
        return None
    dest = audio_dir / "synth_long_english.wav"
    gap = b"\x00\x00" * int(0.4 * 16000)        # 400ms between sentences

    if not dest.exists():
        pcm, params = b"", None
        for row in english:
            src = audio_dir / f"{row['clip_id']}.wav"
            with wave.open(str(src), "rb") as w:
                if params is None:
                    params = w.getparams()
                elif (w.getnchannels(), w.getframerate(), w.getsampwidth()) != \
                        (params.nchannels, params.framerate, params.sampwidth):
                    return None                # refuse to stitch mismatched audio
                pcm += w.readframes(w.getnframes()) + gap
        if params is None:
            return None
        with wave.open(str(dest), "wb") as out:
            out.setnchannels(params.nchannels)
            out.setsampwidth(params.sampwidth)
            out.setframerate(params.framerate)
            out.writeframes(pcm)

    must: list = []
    for row in english:
        for term in (row.get("must_have") or []):
            if term not in must:
                must.append(term)
    return {
        "clip_id": "synth_long_english",
        "gold": " ".join((r.get("gold") or "").strip() for r in english).strip(),
        "must_have": must,
        "language": "English",
        "category": "synth_long_english",
        "trust": "synthetic",
        "audio_ref": {"source_audio": "synth_long_english.wav"},
    }


def _seq(rows, row) -> int:
    """Index of this row within its own category, so sorting interleaves."""
    cat = str(row.get("category"))
    n = 0
    for r in rows:
        if r is row:
            return n
        if str(r.get("category")) == cat:
            n += 1
    return n


if __name__ == "__main__":
    build()
