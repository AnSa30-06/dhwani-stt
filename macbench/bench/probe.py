"""Machine facts and a decode-rate table.

Everything in the sweep is a number of milliseconds, and milliseconds mean
nothing without knowing what this box costs per second of audio. This produces
the two things that make the rest readable:

  --env   what the machine and the installed stack actually are, and which
          backend solution/draft.py resolves to here
  --rtf   seconds of decode per second of audio, per model, measured on a real
          clip through the engine's own transcribe path

RTF is the number the whole design hangs off. A 3.0s final budget on a 10s clip
is generous at RTF 0.08 and already blown at RTF 0.35, and the engine has only
ever been run at the second kind of number.

    python -m bench.probe --env
    python -m bench.probe --rtf large-v3-turbo
    python -m bench.probe --rtf-mix shunyalabs/zero-stt-hinglish
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RTF_CLIP = ROOT / "data/dev/audio/fleurs_hi_in_test_1666.wav"     # 10.2s, Hindi
RTF_CLIP_EN = ROOT / "data/dev/audio/fleurs_en_us_test_1904.wav"  # 10.6s, English


def _pkg(name: str) -> str:
    try:
        mod = __import__(name)
        return str(getattr(mod, "__version__", "installed"))
    except Exception as exc:      # noqa: BLE001
        return f"MISSING ({type(exc).__name__})"


def _sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=15).stdout.strip()
    except Exception:             # noqa: BLE001
        return ""


def _dir_gb(path: Path) -> float:
    try:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return round(total / 1e9, 2)
    except Exception:             # noqa: BLE001
        return -1.0


def env_report() -> dict:
    hf = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache/huggingface"))
    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "packages": {n: _pkg(n) for n in
                     ("numpy", "torch", "transformers", "mlx", "mlx_whisper",
                      "ctranslate2", "faster_whisper", "websockets", "soundfile")},
        "hf_home": str(hf),
        "hf_cache_gb": _dir_gb(hf),
    }
    if sys.platform == "darwin":
        info["mac"] = {
            "sw_vers": _sh("sw_vers", "-productVersion"),
            "chip": _sh("sysctl", "-n", "machdep.cpu.brand_string"),
            "ram_bytes": _sh("sysctl", "-n", "hw.memsize"),
            "perf_cores": _sh("sysctl", "-n", "hw.perflevel0.physicalcpu"),
            "eff_cores": _sh("sysctl", "-n", "hw.perflevel1.physicalcpu"),
            "sandbox_exec": os.path.exists("/usr/bin/sandbox-exec"),
        }
    try:
        import torch
        info["torch_mps"] = bool(getattr(torch.backends, "mps", None)
                                 and torch.backends.mps.is_available())
    except Exception:             # noqa: BLE001
        info["torch_mps"] = None

    sys.path.insert(0, str(ROOT))
    from solution import draft as D
    info["draft"] = {
        "resolved_backend": D._resolve_backend(),
        "final_model": D._model_name(True),
        "draft_model": D._model_name(False),
        "mix_model": D._mix_model(),
        "mix_backend": D._mix_backend(),
        "beam_final": D._beam_size(True),
        "mix_beam": D._mix_beam(),
        "chunk_s": D._chunk_s(),
        "final_budget_s": D._final_budget_s(),
        "speculate": D._speculation_enabled(),
    }
    return info


def _pcm(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as w:
        pcm = w.readframes(w.getnframes())
        return pcm, w.getnframes() / w.getframerate()


def rtf_primary(model: str, backend: str | None = None, clip: Path = RTF_CLIP_EN) -> dict:
    """Load `model` and decode one real clip through the engine's own path."""
    if backend:
        os.environ["DHWANI_BACKEND"] = backend
    os.environ["DHWANI_MODEL"] = model
    sys.path.insert(0, str(ROOT))
    from solution import draft as D

    pcm, secs = _pcm(clip)
    t0 = time.monotonic()
    try:
        D._transcribe(pcm[: int(0.5 * D.BYTES_PER_SEC)], None, "", final=True, fast=True)
    except Exception as exc:      # noqa: BLE001
        return {"model": model, "backend": backend or D._resolve_backend(),
                "error": f"{type(exc).__name__}: {exc}"}
    load_s = time.monotonic() - t0

    timings = []
    for _ in range(2):            # first pass after load can still be warming
        t = time.monotonic()
        words, lang = D._transcribe(pcm, None, "", final=True, fast=True)
        timings.append(time.monotonic() - t)
    decode_s = min(timings)
    return {"model": model, "backend": backend or D._resolve_backend(),
            "clip_s": round(secs, 1), "load_s": round(load_s, 1),
            "decode_s": round(decode_s, 2), "rtf": round(decode_s / secs, 3),
            "beam": D._beam_size(True),
            "text": D._text_from(words, lang)[:120], "lang": lang}


def rtf_mix(repo: str, clip: Path = RTF_CLIP) -> dict:
    os.environ["DHWANI_MIX_MODEL"] = repo
    os.environ.setdefault("DHWANI_MIX_BACKEND", "transformers")
    sys.path.insert(0, str(ROOT))
    from solution import draft as D

    pcm, secs = _pcm(clip)
    t0 = time.monotonic()
    try:
        D._transcribe_mix_transformers(pcm[: int(0.5 * D.BYTES_PER_SEC)])
    except Exception as exc:      # noqa: BLE001
        return {"model": repo, "backend": "transformers",
                "error": f"{type(exc).__name__}: {exc}"}
    load_s = time.monotonic() - t0

    timings = []
    for _ in range(2):
        t = time.monotonic()
        words, _ = D._transcribe_mix_transformers(pcm)
        timings.append(time.monotonic() - t)
    decode_s = min(timings)
    return {"model": repo, "backend": "transformers", "clip_s": round(secs, 1),
            "load_s": round(load_s, 1), "decode_s": round(decode_s, 2),
            "rtf": round(decode_s / secs, 3), "beam": D._mix_beam(),
            "text": D._text_from(words, "hi")[:120]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", action="store_true")
    ap.add_argument("--rtf", help="primary model id/size")
    ap.add_argument("--rtf-mix", help="HF repo for the mix model")
    ap.add_argument("--backend", help="force DHWANI_BACKEND for --rtf")
    ap.add_argument("--out", help="write JSON here as well as stdout")
    args = ap.parse_args()

    if args.env:
        res = env_report()
    elif args.rtf:
        res = rtf_primary(args.rtf, args.backend)
    elif args.rtf_mix:
        res = rtf_mix(args.rtf_mix)
    else:
        ap.error("one of --env / --rtf / --rtf-mix")

    text = json.dumps(res, ensure_ascii=False, indent=1)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
