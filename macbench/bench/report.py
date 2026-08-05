"""Turn the raw results into something readable in one pass.

Writes REPORT.md (everything) and SUMMARY.txt (small enough to paste into a
chat window if sending the folder is awkward).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from streaming_scorecard import end_to_final_points  # noqa: E402

CURVE = "<=1000ms=30/30, 2000ms=24, 3500ms=12, 5000ms=3.6, >5000ms=0"

# How much slower the SCORING host is assumed to be than the machine this ran
# on. builderr score on a frozen M1 Pro; an M4 Pro is roughly twice as quick at
# whisper inference. That gap lands squarely inside the steepest part of the
# curve above — 30 points at 1000 ms, 24 at 2000 — so a configuration tuned to
# the faster machine's clock can look like full marks here and score 24 there.
# Override once the real ratio is known: MACBENCH_SLOWDOWN=1.8
SLOWDOWN = float(os.environ.get("MACBENCH_SLOWDOWN", "2.0"))


def robust_total(row: dict, factor: float = SLOWDOWN) -> float | None:
    """The row's score if every final took `factor` times longer.

    Quality is held constant, which makes this an OPTIMISTIC estimate rather
    than a simulation: on a genuinely slower host some decodes would also miss
    the budget and fall back to worse text, costing quality too. It is still
    the right number to rank on, because the alternative is ranking on a clock
    that is not the one being graded.
    """
    if row.get("quality") is None or not row.get("clips"):
        return None
    lat = [end_to_final_points((c.get("ms") or 0) * factor) for c in row["clips"]]
    if not lat:
        return None
    return round(row["quality"] + sum(lat) / len(lat), 2)


def _load(results: Path) -> tuple[list, dict, list]:
    raw = []
    for f in sorted((results / "raw").glob("*.json")):
        if f.name.endswith(".cfg.json"):
            continue
        try:
            raw.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:          # noqa: BLE001
            pass
    env = {}
    if (results / "env.json").exists():
        env = json.loads((results / "env.json").read_text(encoding="utf-8"))
    rtf = []
    if (results / "rtf.json").exists():
        rtf = json.loads((results / "rtf.json").read_text(encoding="utf-8"))
    return raw, env, rtf


def _fmt(v, nd=2, dash="-"):
    return dash if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def write(results: Path) -> None:
    raw, env, rtf = _load(results)
    by_id = {r.get("id"): r for r in raw}
    control = by_id.get("baseline_stream_slice") or {}
    base_total = control.get("total")

    L: list[str] = ["# dhwani — Apple-hardware measurement run", ""]

    # --- machine
    L += ["## The machine", ""]
    if env:
        mac = env.get("mac") or {}
        ram = mac.get("ram_bytes")
        L += [f"- {mac.get('chip') or env.get('processor') or env.get('machine')}"
              f"  ·  macOS {mac.get('sw_vers', '?')}"
              f"  ·  {env.get('cpu_count')} cores"
              f"  ·  {round(int(ram) / 1e9) if ram and ram.isdigit() else '?'} GB",
              f"- python {env.get('python')}, torch MPS = {env.get('torch_mps')}, "
              f"sandbox-exec = {mac.get('sandbox_exec')}",
              f"- backend solution/draft.py resolved to: "
              f"**{(env.get('draft') or {}).get('resolved_backend')}**",
              f"- HF cache {env.get('hf_cache_gb')} GB at `{env.get('hf_home')}`", ""]
        miss = [k for k, v in (env.get("packages") or {}).items()
                if str(v).startswith("MISSING")]
        if miss:
            L += [f"- **missing packages:** {', '.join(miss)}", ""]

    # --- decode rates
    L += ["## Decode rate (the number everything else is read against)", "",
          "RTF is decode-seconds per audio-second. The final budget defaults to "
          "3.0s; on a 10s clip a whole-buffer final costs `10 x RTF`, so RTF "
          "tells you directly whether the budget ever binds.", "",
          "| model | backend | load s | decode s | clip s | RTF |",
          "|---|---|---|---|---|---|"]
    for r in rtf:
        if r.get("error"):
            L.append(f"| {r.get('model')} | {r.get('backend')} | — | — | — | "
                     f"ERROR: {str(r['error'])[:80]} |")
        else:
            L.append(f"| {r.get('model')} | {r.get('backend')} | {_fmt(r.get('load_s'), 1)} "
                     f"| {_fmt(r.get('decode_s'))} | {_fmt(r.get('clip_s'), 1)} "
                     f"| **{_fmt(r.get('rtf'), 3)}** |")
    L.append("")

    # --- the gate that outranks every other number in this file
    blank = by_id.get("gate_no_blank_finals")
    L += ["## Blank-final gate (read this first)", ""]
    if not blank:
        L += ["Did not run.", ""]
    elif blank.get("error"):
        L += [f"**Could not run:** {str(blank['error'])[:300]}", ""]
    elif blank.get("blank"):
        L += [f"### FAILED — {blank['blank']} of {blank['n']} clips returned an "
              "EMPTY transcript.", "",
              "This is the failure that scored zero on two submitted rounds. "
              "Nothing else in this report means anything until it is fixed. "
              "Per-rung reasons are in `results/logs/gate_no_blank_finals.log`.", ""]
    else:
        L += [f"PASSED — {blank['n']}/{blank['n']} clips returned text on the "
              f"**{blank.get('backend')}** backend.", ""]
        rungs = blank.get("mlx_rungs")
        if rungs:
            L += ["The MLX capability ladder had to step down for these models — "
                  "each entry is a keyword this runtime refused, which on the "
                  "submitted code would have been a blank final:", ""]
            for repo, rung in rungs.items():
                L.append(f"- `{repo}` → rung {rung}")
            L.append("")
        else:
            L += ["The MLX ladder never stepped down: this runtime accepted the "
                  "full argument set.", ""]

    # A blank anywhere else is just as fatal; do not let it hide in a mean.
    blanky = [r for r in raw if r.get("blank")]
    if blanky:
        L += ["### rows with blank finals", ""]
        for r in blanky:
            L.append(f"- `{r.get('id')}` — {r['blank']}/{r.get('n')} clips blank")
        L.append("")

    gate = by_id.get("gate_offline_sandbox")
    if gate:
        L += ["## Offline + sandbox gate", ""]
        if gate.get("error"):
            L += ["**FAILED.** The sealed server could not complete a scored run "
                  "under `sandbox-exec` with `HF_HUB_OFFLINE=1`. Official scoring "
                  "always enforces both, so this outranks every other number here.",
                  "", "```", str(gate.get("stderr") or gate.get("error"))[-2000:], "```", ""]
        else:
            L += [f"PASSED — scored {gate.get('total')} on a 3-clip slice with the "
                  "network blocked and the process sandboxed.", ""]

    # --- baselines
    L += ["## Baseline: the shipped engine", "",
          "| reading | score | quality | latency | median ms | flips | clips |",
          "|---|---|---|---|---|---|---|"]
    for cid, label in (("baseline_evaluator", "real harness (5 runs, sealed server)"),
                       ("baseline_stream", "fast instrument, full corpus"),
                       ("baseline_stream_slice", "fast instrument, sweep slice — the CONTROL")):
        r = by_id.get(cid)
        if not r:
            continue
        if r.get("error"):
            L.append(f"| {label} | ERROR: {str(r['error'])[:70]} | | | | | |")
            continue
        L.append(f"| {label} | **{_fmt(r.get('total'))}** | {_fmt(r.get('quality'))} "
                 f"| {_fmt(r.get('latency'))} | {_fmt(r.get('median_ms'), 0)} "
                 f"| {r.get('flips', r.get('clips_capped', '-'))} | {r.get('n')} |")
    L += ["", f"Latency curve: {CURVE}", ""]

    ev, st = by_id.get("baseline_evaluator"), by_id.get("baseline_stream")
    if ev and st and not ev.get("error") and not st.get("error") \
            and ev.get("total") and st.get("total"):
        L += [f"Calibration: the fast instrument reads **{_fmt(st['total'])}** where "
              f"the real harness reads **{_fmt(ev['total'])}** "
              f"(difference {_fmt(st['total'] - ev['total'])}). Sweep deltas below "
              "are in fast-instrument points and should be read through that gap.", ""]

    # --- sweep
    sweep = [r for r in raw if r.get("mode") == "stream"
             and r.get("id") not in ("baseline_stream", "baseline_stream_slice")]
    ok = [r for r in sweep if not r.get("error") and r.get("total") is not None]
    ok.sort(key=lambda r: -r["total"])
    finals = [r for r in raw if str(r.get("id", "")).startswith("final_")]
    L += ["## Sweep — every knob, on the same 6-clip slice", ""]
    if base_total is not None:
        L += [f"Control (shipped defaults on this slice): **{_fmt(base_total)}**.  "
              "`delta` is versus that.", ""]
    L += ["| config | total | delta | quality | latency | median ms | flips | "
          "paths taken | why it was tried |", "|---|---|---|---|---|---|---|---|---|"]
    for r in ok:
        d = "" if base_total is None else f"{r['total'] - base_total:+.2f}"
        paths = ", ".join(f"{k}:{v}" for k, v in sorted((r.get("paths") or {}).items()))
        L.append(f"| `{r['id']}` | **{_fmt(r['total'])}** | {d} | {_fmt(r.get('quality'))} "
                 f"| {_fmt(r.get('latency'))} | {_fmt(r.get('median_ms'), 0)} "
                 f"| {r.get('flips')} | {paths} | {r.get('why', '')[:110]} |")
    bad = [r for r in sweep if r.get("error")]
    if bad:
        L += ["", "### rows that did not complete", ""]
        for r in bad:
            L.append(f"- `{r['id']}` — {str(r['error'])[:200]}")
    L.append("")

    # --- does the winner survive being run on slower hardware
    L += ["## Hardware headroom — this machine is not the scoring machine", "",
          "builderr score on a frozen **M1 Pro**. Whatever ran this is very "
          "likely faster, and the latency curve is at its steepest between "
          "1000 ms (30/30) and 3500 ms (12/30) — so a configuration tuned to "
          "this machine's clock can read full marks here and lose six points "
          "there. Each column re-scores the same run with every final taking "
          "that much longer. **Pick a configuration that holds up across the "
          "row, not one that wins the 1.0x column.**", "",
          "Quality is held fixed, so these are optimistic: on a slower host "
          "some decodes would also miss the budget and return worse text.", "",
          "| config | 1.0x | 1.5x | 2.0x | 2.5x | 3.0x |", "|---|---|---|---|---|---|"]
    for r in ([control] if control else []) + ok[:6] + finals:
        if r.get("quality") is None or not r.get("clips"):
            continue
        cells = " | ".join(_fmt(robust_total(r, f)) for f in (1.0, 1.5, 2.0, 2.5, 3.0))
        L.append(f"| `{r.get('id')}` | {cells} |")
    L += ["", f"Ranking of the finalists used the **{SLOWDOWN}x** column "
          "(MACBENCH_SLOWDOWN). Once the RTF table above can be compared "
          "against a known M1 Pro figure, re-run `python -m bench.report` with "
          "the real ratio to re-read every row.", ""]

    # --- per category for the interesting rows
    L += ["## Per category (control and the top five)", "",
          "| config | english q/lat | hindi q/lat | hinglish q/lat |",
          "|---|---|---|---|"]
    for r in ([control] if control else []) + ok[:5]:
        cats = r.get("by_category") or {}

        def cell(name):
            b = cats.get(name)
            return "-" if not b else f"{b['quality']:.1f} / {b['latency']:.1f}"
        L.append(f"| `{r.get('id')}` | {cell('fleurs_english')} | "
                 f"{cell('fleurs_hindi')} | {cell('openslr104_hinglish')} |")
    L.append("")

    # --- finalists
    L += ["## Finalists — the real harness, full corpus, 5 runs per clip", ""]
    if not finals:
        L += ["None ran. Either nothing beat the control on the fast instrument "
              "(itself a finding: the shipped configuration is a local optimum on "
              "this hardware) or the run stopped before this stage.", ""]
    else:
        L += ["| config | score /100 | median ms | meaning | WER | capped | env |",
              "|---|---|---|---|---|---|---|"]
        for r in finals:
            if r.get("error"):
                L.append(f"| `{r['id']}` | ERROR: {str(r['error'])[:70]} | | | | | |")
                continue
            L.append(f"| `{r['id']}` | **{_fmt(r.get('total'))}** "
                     f"| {_fmt(r.get('median_ms'), 0)} | {_fmt(r.get('meaning_mean'), 3)} "
                     f"| {_fmt(r.get('wer_mean'), 3)} "
                     f"| {r.get('clips_capped')}/{r.get('n')} "
                     f"| `{json.dumps(r.get('env'))}` |")
        L.append("")

    # Anything not printed above. A row that quietly vanishes from a report
    # reads as "covered everything" when it is really "this one broke".
    shown = {"baseline_evaluator", "baseline_stream", "baseline_stream_slice",
             "gate_offline_sandbox"}
    shown |= {r.get("id") for r in sweep} | {r.get("id") for r in finals}
    other = [r for r in raw if r.get("id") not in shown]
    if other:
        L += ["## Other rows", ""]
        for r in other:
            L.append(f"- `{r.get('id')}` [{r.get('mode')}] — "
                     f"{'ERROR: ' + str(r['error'])[:200] if r.get('error') else 'total ' + _fmt(r.get('total'))}")
        L.append("")

    # --- per-clip detail, because a mean hides the clip that scored zero
    L += ["## Per-clip detail (control)", ""]
    if control.get("clips"):
        L += ["| clip | category | quality | latency | ms | path | covered | flip |",
              "|---|---|---|---|---|---|---|---|"]
        for c in control["clips"]:
            L.append(f"| {c['clip_id'][:34]} | {c['category']} | {_fmt(c['quality'])} "
                     f"| {_fmt(c['latency'])} | {c['ms']} | {c.get('path') or '-'} "
                     f"| {c.get('covered')} | {'YES' if c.get('flip') else ''} |")
        L.append("")

    (results / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # --- the paste-into-chat version
    S = ["dhwani macbench summary", "=" * 40]
    if blank:
        if blank.get("error"):
            S.append(f"BLANK-FINAL GATE: could not run — {str(blank['error'])[:90]}")
        elif blank.get("blank"):
            S.append(f"BLANK-FINAL GATE: *** FAILED *** {blank['blank']}/{blank['n']} "
                     "clips returned nothing — read no further, this is the bug")
        else:
            S.append(f"BLANK-FINAL GATE: passed ({blank['n']}/{blank['n']} clips have "
                     f"text, backend {blank.get('backend')}, "
                     f"mlx rungs {blank.get('mlx_rungs') or 'none needed'})")
    for r in blanky:
        S.append(f"  !! {r.get('id')}: {r['blank']}/{r.get('n')} BLANK")
    if env:
        mac = env.get("mac") or {}
        S.append(f"machine: {mac.get('chip') or env.get('machine')} / macOS "
                 f"{mac.get('sw_vers','?')} / backend "
                 f"{(env.get('draft') or {}).get('resolved_backend')}")
    for r in rtf:
        if not r.get("error"):
            S.append(f"rtf  {r.get('model'):34s} {r.get('backend'):12s} "
                     f"{r.get('rtf')}  ({r.get('decode_s')}s for {r.get('clip_s')}s)")
    if gate:
        S.append(f"offline+sandbox gate: {'FAILED — ' + str(gate.get('error'))[:120] if gate.get('error') else 'passed'}")
    for cid in ("baseline_evaluator", "baseline_stream", "baseline_stream_slice"):
        r = by_id.get(cid)
        if r and not r.get("error"):
            S.append(f"{cid:24s} total {_fmt(r.get('total')):>7s}  q {_fmt(r.get('quality')):>6s}"
                     f"  lat {_fmt(r.get('latency')):>6s}  median {_fmt(r.get('median_ms'),0)}ms")
    S.append("-" * 40)
    for r in ok:
        d = "" if base_total is None else f"{r['total'] - base_total:+6.2f}"
        S.append(f"{r['id']:30s} {_fmt(r['total']):>7s} {d}  "
                 f"q{_fmt(r.get('quality')):>6s} lat{_fmt(r.get('latency')):>6s} "
                 f"{_fmt(r.get('median_ms'),0):>6s}ms flips{r.get('flips')}")
    for r in bad:
        S.append(f"{r['id']:30s} FAILED  {str(r['error'])[:80]}")
    S.append("-" * 40)
    for r in finals:
        S.append(f"{r['id']:30s} {_fmt(r.get('total')):>7s} "
                 f"{_fmt(r.get('median_ms'),0)}ms  env={json.dumps(r.get('env'))}")
    for r in other:
        S.append(f"{str(r.get('id')):30s} [{r.get('mode')}] "
                 f"{'FAILED ' + str(r['error'])[:70] if r.get('error') else _fmt(r.get('total'))}")
    (results / "SUMMARY.txt").write_text("\n".join(S) + "\n", encoding="utf-8")
    print("\n".join(S))


if __name__ == "__main__":
    write(Path(__file__).resolve().parents[1] / "results")
