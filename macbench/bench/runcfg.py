"""Run exactly ONE config and write its result. Always a fresh process.

A fresh process per config is not tidiness, it is correctness. solution/draft.py
resolves its backend once and caches it, caches every loaded model in a
module-global dict, and decides at IMPORT time whether to warm on a background
thread. Sweeping DHWANI_BACKEND or DHWANI_MODEL inside one interpreter would
measure whichever one happened to load first. The price is one model load per
row (~10-20s warm from the HF cache) and it buys a result that means what it says.

    python -m bench.runcfg <config.json> <out.json>
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _apply_env(env: dict) -> None:
    """Set the config's environment BEFORE solution.draft is imported."""
    for k, v in env.items():
        if v == "":
            os.environ[k] = ""      # deliberate: "" disables the mix model
        else:
            os.environ[k] = str(v)


def _take_slice(clips: list, want: dict | None) -> list:
    if not want:
        return clips
    out, taken = [], {}
    for c in clips:
        cat = c["category"]
        if taken.get(cat, 0) >= want.get(cat, 0):
            continue
        taken[cat] = taken.get(cat, 0) + 1
        out.append(c)
    return out


# --- mode: stream — the cheap instrument, in this process -------------------

def run_stream(cfg: dict) -> dict:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts"))
    import stream_latency as SL

    from solution import draft as D

    t0 = time.monotonic()
    D.warm_models()
    warm_s = time.monotonic() - t0

    clips = _take_slice(SL.load_clips(), cfg.get("slice"))
    rows = []
    for clip in clips:
        runs = [SL.stream_once(D, clip["pcm"]) for _ in range(cfg.get("runs", 2))]
        runs.sort(key=lambda r: r["ms"])
        median = runs[(len(runs) - 1) // 2]      # the median-LATENCY run, as scored
        ms = statistics.median([r["ms"] for r in runs])
        pts, meaning, flip = SL.quality_points(clip, median["text"])
        rows.append({
            "clip_id": clip["clip_id"], "category": clip["category"],
            "quality": pts, "latency": round(SL.end_to_final_points(ms), 2),
            "ms": round(ms), "ms_all": [round(r["ms"]) for r in runs],
            "path": median["path"], "meaning": meaning, "flip": flip,
            "covered": round(median["covered"], 2),
            "spec_covered": round(median["spec_covered"], 2),
            "audio_s": round(median["audio_s"], 1), "pred": median["text"],
        })
        print(f"   {clip['clip_id'][:30]:32s} q{pts:5.1f} +lat{rows[-1]['latency']:5.1f}"
              f" = {pts + rows[-1]['latency']:5.1f}  {ms:6.0f}ms  {median['path'] or '-':12s}"
              f" cov{median['covered']:4.0%}{'  FLIP' if flip else ''}", flush=True)

    n = len(rows) or 1
    quality = sum(r["quality"] for r in rows) / n
    latency = sum(r["latency"] for r in rows) / n
    paths: dict[str, int] = {}
    for r in rows:
        paths[r["path"] or "-"] = paths.get(r["path"] or "-", 0) + 1
    return {
        "quality": round(quality, 2), "latency": round(latency, 2),
        "total": round(quality + latency, 2),
        "median_ms": round(statistics.median([r["ms"] for r in rows])) if rows else None,
        "flips": sum(r["flip"] for r in rows), "n": len(rows),
        # The number that decides whether anything else here is worth reading.
        # Two submitted rounds scored zero on exactly this and nobody local
        # could see it, so it is now reported first-class on every single row.
        "blank": sum(1 for r in rows if not (r.get("pred") or "").strip()),
        "backend": D._resolve_backend(),
        "mlx_rungs": dict(D._MLX_LEVEL),
        "paths": paths, "warm_s": round(warm_s, 1),
        "by_category": _by_category(rows), "clips": rows,
    }


def _by_category(rows: list) -> dict:
    out: dict[str, dict] = {}
    for r in rows:
        b = out.setdefault(r["category"], {"n": 0, "quality": 0.0, "latency": 0.0,
                                           "flips": 0, "ms": []})
        b["n"] += 1
        b["quality"] += r["quality"]
        b["latency"] += r["latency"]
        b["flips"] += int(r["flip"])
        b["ms"].append(r["ms"])
    for b in out.values():
        b["quality"] = round(b["quality"] / b["n"], 2)
        b["latency"] = round(b["latency"] / b["n"], 2)
        b["total"] = round(b["quality"] + b["latency"], 2)
        b["median_ms"] = round(statistics.median(b["ms"]))
        del b["ms"]
    return out


# --- mode: evaluator — the real harness, sealed server and all --------------

def run_evaluator(cfg: dict) -> dict:
    """The ground truth. Launches solution/stream_server.py exactly as builderr
    does, including the READY deadline and the cold model load on clip one."""
    from bench import localset
    manifest = localset.build(verbose=False)

    cmd = [sys.executable, str(ROOT / "evaluator.py"),
           "--manifest", str(manifest), "--runs", str(cfg.get("runs", 5)), "--json"]
    if not cfg.get("offline"):
        cmd.append("--no-offline")
    want = cfg.get("slice")
    if want:
        cmd += ["--per-category", str(max(want.values()))]
    cmd += ["--server-log", str(ROOT / "macbench/results/raw" /
                               f"server-{cfg['id']}.log")]

    print(f"   $ {' '.join(cmd[1:])}", flush=True)
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env)
    if proc.returncode != 0:
        return {"error": "evaluator exited non-zero",
                "returncode": proc.returncode,
                "stderr": proc.stderr[-4000:], "stdout": proc.stdout[-4000:]}
    try:
        start = proc.stdout.index("{")
        res = json.loads(proc.stdout[start:])
    except Exception as exc:      # noqa: BLE001
        return {"error": f"could not parse evaluator JSON: {exc}",
                "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}

    # Fold it into the same shape the stream mode returns, so one report can
    # print both. The evaluator does not split the axes, so quality is the
    # residual — which is exactly how the 62.51 was decomposed in the first place.
    res["total"] = res.get("overall_score")
    res["median_ms"] = res.get("median_end_to_final_ms")

    # Split the axes. The evaluator reports one number per clip; the latency
    # half is recoverable because the published curve is a pure function of the
    # clip's own median end-to-final, and quality is then the residual. This is
    # the decomposition that showed 62.51 was ~49 quality + ~13 latency — i.e.
    # that the axis nobody could measure was the one with the room in it.
    sys.path.insert(0, str(ROOT))
    from streaming_scorecard import end_to_final_points
    clips = res.get("clips") or []
    lat = [end_to_final_points(c.get("median_end_to_final_ms") or 0.0) for c in clips]
    if lat:
        res["latency"] = round(sum(lat) / len(lat), 2)
        res["quality"] = round((res["total"] or 0.0) - res["latency"], 2)
        for c, p in zip(clips, lat):
            c["latency_points"] = round(p, 2)
    return res


def main() -> None:
    cfg_path, out_path = sys.argv[1], sys.argv[2]
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    _apply_env(cfg.get("env", {}))

    print(f"== {cfg['id']} [{cfg['mode']}] runs={cfg.get('runs')} "
          f"env={cfg.get('env') or '(shipped defaults)'}", flush=True)

    t0 = time.monotonic()
    try:
        res = run_evaluator(cfg) if cfg["mode"] == "evaluator" else run_stream(cfg)
    except Exception as exc:      # noqa: BLE001
        import traceback
        traceback.print_exc()
        res = {"error": f"{type(exc).__name__}: {exc}"}
    res["id"] = cfg["id"]
    res["mode"] = cfg["mode"]
    res["why"] = cfg.get("why", "")
    res["env"] = cfg.get("env", {})
    res["runs"] = cfg.get("runs")
    res["wall_s"] = round(time.monotonic() - t0, 1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n",
                              encoding="utf-8")
    print(f"== {cfg['id']} done in {res['wall_s']}s -> total "
          f"{res.get('total')}  {res.get('error') or ''}", flush=True)


if __name__ == "__main__":
    main()
