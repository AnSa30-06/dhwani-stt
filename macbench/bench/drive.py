"""The battery. Runs every config, survives anything, and can be re-run.

Design rules, all of them learned the hard way on the Windows box:

  * one SUBPROCESS per config, with a hard timeout — a wedged decode kills its
    own row, never the run;
  * every result is written the moment it lands, and an existing result is
    skipped on the next invocation, so a laptop that sleeps costs one row;
  * a row that fails records WHY and the battery continues. A three-hour run
    that aborts on row four is worth nothing;
  * the sweep rows are cheap and comparative; the finalists are expensive and
    faithful. Nothing gets the expensive treatment until the cheap pass says
    it is worth it.

    python -m bench.drive              # everything (~2.5-3.5h)
    QUICK=1 python -m bench.drive      # the rows most likely to move it (~45m)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "macbench"
RESULTS = BENCH / "results"
RAW = RESULTS / "raw"
LOGS = RESULTS / "logs"

sys.path.insert(0, str(BENCH))
from bench import configs as C          # noqa: E402
from bench import localset              # noqa: E402
from bench import report as R           # noqa: E402

MAX_HOURS = float(os.environ.get("MACBENCH_MAX_HOURS", "6"))
START = time.monotonic()


def say(msg: str) -> None:
    mins = (time.monotonic() - START) / 60
    print(f"\n[{mins:5.1f}m] {msg}", flush=True)


def _hours_left() -> float:
    return MAX_HOURS - (time.monotonic() - START) / 3600


def _spawn(cmd: list, env: dict, log: Path, timeout_s: float) -> bool:
    """Run a child, tee its output live, and ENFORCE the timeout.

    The obvious shape — read the pipe to EOF, then wait(timeout=...) — cannot
    time anything out: a wedged child never closes its pipe, so the read loop
    blocks forever and the wait is never reached. A row that hangs on one
    model's download would otherwise eat the entire run. So the reader lives on
    its own thread and the parent waits on the PROCESS.

    Returns True if it was killed for overrunning.
    """
    import threading

    proc = subprocess.Popen(cmd, cwd=str(BENCH), env=env, text=True,
                            encoding="utf-8", errors="replace",
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1)

    def _tee() -> None:
        with open(log, "w", encoding="utf-8") as fh:
            for line in proc.stdout:            # type: ignore[union-attr]
                fh.write(line)
                fh.flush()
                if line.startswith(("   ", "== ")):
                    print(line.rstrip(), flush=True)

    reader = threading.Thread(target=_tee, daemon=True)
    reader.start()
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)
        reader.join(timeout=5)
        return True
    reader.join(timeout=10)
    return False


# --- one config -------------------------------------------------------------

def run_config(cfg: dict, force: bool = False) -> dict:
    out = RAW / f"{cfg['id']}.json"
    if out.exists() and not force:
        try:
            done = json.loads(out.read_text(encoding="utf-8"))
            if "error" not in done:
                say(f"skip {cfg['id']} (already done: total {done.get('total')})")
                return done
        except Exception:          # noqa: BLE001
            pass                   # unreadable result: rerun it

    cfg_path = RAW / f"{cfg['id']}.cfg.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False), encoding="utf-8")

    say(f"run  {cfg['id']} — {cfg['why'][:90]}")
    log = LOGS / f"{cfg['id']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(BENCH), PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "-m", "bench.runcfg", str(cfg_path), str(out)]

    timed_out = _spawn(cmd, env, log, cfg.get("timeout_s", 1200))
    if timed_out:
        res = {"id": cfg["id"], "mode": cfg["mode"], "env": cfg.get("env", {}),
               "why": cfg.get("why", ""),
               "error": f"timeout after {cfg.get('timeout_s')}s"}
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        say(f"TIMEOUT {cfg['id']} — recorded and moving on")
        return res

    if not out.exists():
        res = {"id": cfg["id"], "mode": cfg["mode"], "env": cfg.get("env", {}),
               "why": cfg.get("why", ""),
               "error": f"runcfg produced no result (see {log.name})"}
        out.write_text(json.dumps(res, indent=1), encoding="utf-8")
        say(f"FAILED {cfg['id']} — see results/logs/{log.name}")
        return res
    return json.loads(out.read_text(encoding="utf-8"))


# --- stage 0: facts about the machine ---------------------------------------

def stage_probe() -> None:
    say("stage 0/4 — machine facts and decode rates")
    envfile = RESULTS / "env.json"
    if not envfile.exists():
        try:
            subprocess.run([sys.executable, "-m", "bench.probe", "--env",
                            "--out", str(envfile)], cwd=str(BENCH), timeout=600,
                           env=dict(os.environ, PYTHONPATH=str(BENCH)))
        except Exception as exc:      # noqa: BLE001
            say(f"   env probe failed ({exc}) — continuing without it")

    rtf_file = RESULTS / "rtf.json"
    if rtf_file.exists():
        return
    rows = []
    # If the blank gate's warm converted the mix model, measure the converted
    # copy too — that number IS the payoff of the submitted build's riskiest
    # change, and only a real Apple machine can produce it.
    mix_specs = [("--rtf-mix", "shunyalabs/zero-stt-hinglish", None)]
    try:
        sys.path.insert(0, str(ROOT))
        from solution import draft as _D
        conv = _D._mix_mlx_dir(convert=False)
        if conv:
            mix_specs.append(("--rtf", conv, None))
    except Exception:          # noqa: BLE001 — probe rows are best-effort
        pass
    # One subprocess per model: loading turbo + large-v3 + medium + a 3GB mix
    # model into one interpreter is how an 8GB Mac gets OOM-killed halfway.
    for spec in [("--rtf", "large-v3-turbo", None), ("--rtf", "medium", None),
                 ("--rtf", "small", None), ("--rtf", "large-v3", None),
                 ("--rtf", "large-v3-turbo", "ctranslate2"),
                 *mix_specs]:
        flag, model, backend = spec
        cmd = [sys.executable, "-m", "bench.probe", flag, model]
        if backend:
            cmd += ["--backend", backend]
        say(f"        decode rate: {model}{' via ' + backend if backend else ''}")
        try:
            proc = subprocess.run(cmd, cwd=str(BENCH), capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=2400,
                                  env=dict(os.environ, PYTHONPATH=str(BENCH)))
            rows.append(json.loads(proc.stdout[proc.stdout.index("{"):]))
        except subprocess.TimeoutExpired:
            rows.append({"model": model, "backend": backend,
                         "error": "timed out (download or decode took over 40 min)"})
        except Exception:          # noqa: BLE001
            rows.append({"model": model, "backend": backend,
                         "error": ((proc.stderr or proc.stdout)[-800:]
                                   if "proc" in dir() else "probe did not start")})
        print(f"        -> {json.dumps(rows[-1], ensure_ascii=False)[:200]}", flush=True)
    rtf_file.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


def stage_selftest() -> None:
    """The suite has two failures on Windows that are supposed to be artifacts
    of Windows. On a Mac one of them (the sandbox profile) must pass, and if it
    does not, that is a submission-breaking finding, not a test nit."""
    out = RESULTS / "pytest.txt"
    if out.exists():
        return
    say("stage 0/4 — test suite on this machine")
    try:
        proc = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q"],
                              cwd=str(ROOT), capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=1800)
        out.write_text(proc.stdout[-20000:] + "\n" + proc.stderr[-4000:], encoding="utf-8")
        print(proc.stdout[-1500:], flush=True)
    except Exception as exc:          # noqa: BLE001
        out.write_text(f"pytest did not complete: {exc}\n", encoding="utf-8")
        say(f"   test suite did not complete ({exc}) — continuing")


def stage_blank_gate() -> bool:
    """THE go/no-go. Does the shipped engine produce TEXT on this machine?

    builderr ran two submitted revisions and could publish neither: "the default
    MLX path requests beam search, which the scoring runtime does not support,
    so every final came back blank." Weeks of quality tuning were graded zero,
    and nothing available here could see it, because no machine here could run
    the MLX path at all.

    So this runs first, on real Apple silicon, through the real backend, and
    asks the only question that gates the rest: did every clip come back with
    words in it. Everything else in this battery is worth exactly nothing until
    this passes.
    """
    gate = {"id": "gate_no_blank_finals", "mode": "stream", "runs": 1,
            "slice": {"fleurs_english": 1, "fleurs_hindi": 1,
                      "openslr104_hinglish": 1},
            "timeout_s": 1800, "env": {},
            "why": "does the shipped engine return TEXT on real Apple silicon"}
    res = run_config(gate)
    blanks = res.get("blank")
    if res.get("error") or blanks is None:
        say("!! BLANK-FINAL GATE could not run — treat everything below as "
            "unverified. See results/raw/gate_no_blank_finals.json")
        return False
    if blanks:
        say(f"!! BLANK-FINAL GATE FAILED — {blanks} of {res.get('n')} clips "
            "returned an EMPTY transcript. This is the failure that scored zero "
            "twice. Nothing else in this run matters until it is fixed; the "
            "per-rung reasons are in results/logs/gate_no_blank_finals.log")
        return False
    say(f"   blank-final gate PASSED — {res.get('n')}/{res.get('n')} clips "
        f"returned text on the {res.get('backend')} backend "
        f"(rungs used: {res.get('mlx_rungs') or 'rung 0, nothing refused'})")
    mix_mlx = res.get("mix_mlx")
    if mix_mlx:
        say(f"   ** MIX MODEL IS ON MLX ** (converted + decode-verified at {mix_mlx}) "
            "— the submitted build's biggest unknown is CONFIRMED on this machine")
    else:
        say("   mix model is on transformers (conversion absent or failed its "
            "decode verify — see the warm log lines above; the engine still "
            "works, Indic finals just run ~1s slower)")
    return True


def stage_offline_gate() -> None:
    """THE risk check. Official scoring runs the sealed server under
    sandbox-exec with HF_HUB_OFFLINE=1. If model loading needs the network on
    first touch, or the sandbox blocks a path the loader reads, the submission
    scores zero and nothing in the quality sweep matters. No Apple machine has
    ever run this."""
    gate = {"id": "gate_offline_sandbox", "mode": "evaluator", "runs": 1,
            "offline": True, "slice": {"fleurs_english": 1, "fleurs_hindi": 1,
                                        "openslr104_hinglish": 1},
            "timeout_s": 1800, "env": {},
            "why": "does the SEALED server survive sandbox-exec + offline HF on a real Mac"}
    res = run_config(gate)
    if res.get("error"):
        say("!! OFFLINE/SANDBOX GATE FAILED — this is the most important line in "
            "the whole run. See results/raw/gate_offline_sandbox.json")
    else:
        say(f"   offline+sandbox gate PASSED (score {res.get('total')})")


# --- stages 1-3: sweep, then confirm ----------------------------------------

def stage_sweep(rows: list) -> list:
    say(f"stage 1/4 — {len(rows)} configs on the fast instrument")
    out = []
    for cfg in rows:
        # Stop early enough that the confirmation stage still fits. A sweep row
        # is a ranking; a finalist is a NUMBER. Spending the last hour on more
        # rankings and never confirming one would be the wrong trade.
        if _hours_left() <= 1.5:
            say(f"time budget nearly spent — skipping {cfg['id']} and the rest of "
                "the sweep so the finalists still run")
            break
        out.append(run_config(cfg))
    return out


def pick_finalists(results: list, baseline_total: float | None) -> list:
    """The cheap pass ranks; the expensive pass confirms. Only rows that beat
    the control on the SAME slice are worth a faithful run."""
    scored = [r for r in results
              if r.get("mode") == "stream" and not r.get("error")
              and r.get("total") is not None
              and r.get("id") not in ("baseline_stream", "baseline_stream_slice")]
    if baseline_total is not None:
        scored = [r for r in scored if r["total"] > baseline_total]
    # Rank on ROBUSTNESS, not on this machine's clock. A row that wins here by
    # shaving 200ms off an already-fast final wins nothing on a host where that
    # final takes twice as long; a row that wins by needing less decoding wins
    # on both. Summing the two readings prefers the second kind.
    scored.sort(key=lambda r: -((r["total"] or 0)
                                + (R.robust_total(r) or r["total"] or 0)))
    finalists = []
    for r in scored[:2]:
        finalists.append({
            "id": f"final_{r['id']}", "mode": "evaluator", "runs": 5,
            "slice": None, "timeout_s": 5400, "env": dict(r["env"]),
            "why": f"top sweep row ({r['total']} vs control {baseline_total}) "
                   f"confirmed on the real harness, full corpus",
        })

    # And the composition. Stacking independently-good knobs is a classic way to
    # be wrong — knobs interact — which is precisely why it is MEASURED here
    # rather than assumed and shipped.
    stack: dict = {}
    for r in scored[:6]:
        for k, v in r["env"].items():
            stack.setdefault(k, v)
    if len(stack) > 1 and not any(f["env"] == stack for f in finalists):
        finalists.append({
            "id": "final_stacked", "mode": "evaluator", "runs": 5, "slice": None,
            "timeout_s": 5400, "env": stack,
            "why": "every knob that beat the control, applied together — a "
                   "hypothesis about interaction, not a recommendation",
        })
    return finalists


def stage_finalists(results: list) -> list:
    baseline = next((r for r in results if r.get("id") == "baseline_stream_slice"
                     and not r.get("error")), None)
    base_total = baseline.get("total") if baseline else None
    finalists = pick_finalists(results, base_total)
    if not finalists:
        say("stage 2/4 — nothing beat the control on the fast instrument. That is "
            "a result: the shipped configuration is a local optimum on this "
            "hardware and the gap is elsewhere.")
        return []
    say(f"stage 2/4 — {len(finalists)} finalists on the REAL harness, full corpus")
    out = []
    for cfg in finalists:
        if _hours_left() <= 0.4:
            say(f"out of time before {cfg['id']} — recorded as not run")
            continue
        out.append(run_config(cfg))
    return out


def main() -> None:
    for d in (RESULTS, RAW, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    say(f"dhwani macbench — budget {MAX_HOURS}h, results in {RESULTS}")
    localset.build()

    smoke = os.environ.get("SMOKE") == "1"
    if smoke:
        say("SMOKE=1 — plumbing check only: no model comparison, no gate, no finalists")
    else:
        stage_blank_gate()      # first: everything else is moot if this fails
        stage_probe()
        stage_selftest()
        stage_offline_gate()

    if os.environ.get("SMOKE") == "1":
        rows = C.smoke_subset()
    elif os.environ.get("QUICK") == "1":
        rows = C.quick_subset()
    else:
        rows = C.ALL
    if os.environ.get("SKIP_BIG_MODELS") == "1":
        dropped = [r["id"] for r in rows if r["id"] in C.BIG_MODEL_ROWS]
        rows = [r for r in rows if r["id"] not in C.BIG_MODEL_ROWS]
        # Say what was dropped. A run that silently covers less than it claims
        # reads afterwards as "we tried everything".
        say(f"SKIP_BIG_MODELS=1 — not running {len(dropped)} multi-GB model rows: "
            f"{', '.join(dropped)}")
    only = os.environ.get("MACBENCH_ONLY")
    if only:
        want = {s.strip() for s in only.split(",")}
        rows = [r for r in rows if r["id"] in want]
        say(f"MACBENCH_ONLY set — running {len(rows)} of the matrix")

    results = stage_sweep(rows)
    finals = [] if os.environ.get("SMOKE") == "1" else stage_finalists(results)

    say("stage 3/4 — writing the report")
    R.write(RESULTS)
    say(f"stage 4/4 — DONE. {len(results)} sweep rows, {len(finals)} finalists.")
    print(f"\n  Send back the whole folder:  {RESULTS}\n", flush=True)


if __name__ == "__main__":
    main()
