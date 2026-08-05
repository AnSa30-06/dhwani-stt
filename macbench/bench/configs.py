"""The experiment matrix.

Every row is one question that could not be answered on the machine this engine
was written on. That box decodes 10-20x slower than the scoring host, which
means the 30-point LATENCY axis — nearly a third of the card — has never once
been read. Three rounds of tuning went into the 70-point quality axis because
it was the only one visible, and quality has been flat at ~58/70 across five
model swaps. The gap to 82.41 is almost certainly on the axis nobody measured.

Read `why` on each row as the hypothesis. A row that comes back neutral is a
result: it closes a branch instead of leaving it as a maybe.
"""
from __future__ import annotations

# Categories in the local corpus, and how many clips of each the fast sweep
# uses. The full set is 15 clips / ~146s; at 1x real time a full pass costs
# ~3.5 minutes, so a 30-row sweep over all of them would run most of a day.
# The slice keeps every category represented — English, pure Hindi and Hinglish
# take three genuinely different paths through _decode_final_segment.
SWEEP_SLICE = {"fleurs_english": 2, "fleurs_hindi": 2, "openslr104_hinglish": 2}


def cfg(cid, why, env, *, mode="stream", runs=2, slice_=True, timeout_s=1200):
    return {"id": cid, "why": why, "env": env, "mode": mode, "runs": runs,
            "slice": SWEEP_SLICE if slice_ else None, "timeout_s": timeout_s}


# --- baselines --------------------------------------------------------------
# Two readings of the SAME engine: one through the real evaluator (the sealed
# server, 5 runs, median latency, cold start included) and one through the cheap
# in-process instrument the sweep uses. The pair is a calibration: it says how
# much a sweep delta is worth in evaluator points. Without it every number below
# is a relative ranking with no scale.

BASELINES = [
    cfg("baseline_evaluator", "the real harness on the shipped engine — the number "
        "directly comparable to the 62.51 builderr reported",
        {}, mode="evaluator", runs=5, slice_=False, timeout_s=3600),
    cfg("baseline_stream", "the same engine through the sweep's own instrument, so "
        "sweep deltas can be converted into evaluator points",
        {}, runs=3, slice_=False, timeout_s=2400),
    cfg("baseline_stream_slice", "and on the 6-clip slice every sweep row uses — the "
        "control every delta below is measured against",
        {}, runs=2),
]


# --- the latency axis -------------------------------------------------------

LATENCY = [
    # The budget default REVERSED after the first Apple-hardware run: 3.0 -> 5.4.
    # The same clip on the same machine scored 83.2 when its decode finished at
    # 2512ms and 24.7 when it landed at 3005ms and hit the fallback. The caps
    # are on the TOTAL and they are gentle (70 past 4s, 50 past 6s), so a late
    # real decode is worth far more than a punctual fragment. These rows exist
    # to check that reversal rather than assume it.
    cfg("budget_3.0", "the OLD default, now a control: how much did the tight "
        "budget cost", {"DHWANI_FINAL_BUDGET_S": "3.0"}),
    cfg("budget_2.0", "tighter still — should be clearly worse if the reversal is right",
        {"DHWANI_FINAL_BUDGET_S": "2.0"}),
    cfg("budget_7.0", "past the 6000ms cap edge, where a clip caps at 50. Should "
        "be worse than 5.4 — if it is not, the cap is not biting and the budget "
        "can go further still", {"DHWANI_FINAL_BUDGET_S": "7.0"}),
    cfg("be_cover_0", "trust the fallback text at any coverage, i.e. the old "
        "behaviour of returning rolling partials on time",
        {"DHWANI_BE_MIN_COVER": "0"}),

    # Chunking was rejected at -6.90/70 on quality, but that was measured where
    # the LATENCY half could not be seen at all. The rejection may not survive
    # contact with the real clock.
    cfg("chunk_0", "no committer at all — a clean whole-buffer final every time",
        {"DHWANI_CHUNK_S": "0"}),
    cfg("chunk_6", "the rejected setting, re-measured with the latency axis visible",
        {"DHWANI_CHUNK_S": "6"}),
    cfg("chunk_8", "between the rejected 6 and the shipped 12", {"DHWANI_CHUNK_S": "8"}),
    cfg("chunk_6_nopin", "chunk 6 with per-window language re-detection. The -6.90 "
        "regression was blamed on _fc_lang pinning from the first window on "
        "code-switched audio; this separates the two for the first time",
        {"DHWANI_CHUNK_S": "6", "DHWANI_FC_LANG_PIN": "0"}),

    cfg("speculate_off", "how much the speculator is actually worth — it is the whole "
        "reason the engine is shaped this way and has never been priced",
        {"DHWANI_SPECULATE": "0"}),

    # Partials are worth ZERO points by the published protocol and are the only
    # other thing competing for the accelerator.
    cfg("partials_off", "stop spending the GPU on transcripts that cannot score",
        {"DHWANI_PARTIALS": "0"}),

    # The new work. A speculation overtaken by speech is thrown away whole
    # today, and a speaker who never pauses gets no speculation at all.
    cfg("spec_join", "keep the prefix of an overtaken speculation and decode only "
        "what it missed", {"DHWANI_SPEC_JOIN": "1"}),
    cfg("spec_periodic_4", "arm a speculation every 4s even without a pause",
        {"DHWANI_SPEC_PERIODIC_S": "4"}),
    cfg("spec_periodic_2", "and every 2s — more coverage, more GPU contention",
        {"DHWANI_SPEC_PERIODIC_S": "2"}),
    cfg("spec_join_periodic_4", "the pair, which is the real proposal: a rolling "
        "whole-buffer decode plus a cheap join. This is the chunked final's "
        "latency win WITHOUT the segment decodes that cost it -6.90/70",
        {"DHWANI_SPEC_JOIN": "1", "DHWANI_SPEC_PERIODIC_S": "4"}),
    cfg("spec_join_periodic_2_nopartials", "the whole idea at full strength",
        {"DHWANI_SPEC_JOIN": "1", "DHWANI_SPEC_PERIODIC_S": "2",
         "DHWANI_PARTIALS": "0"}),
    cfg("spec_silence_015", "arm on half as much trailing silence — more clips get a "
        "speculation, some of them stale", {"DHWANI_SPEC_SILENCE_S": "0.15"}),

    # Pure Hindi is the only path that pays for two models in series.
    cfg("mix_parallel", "overlap the primary and mix decodes so pure Hindi costs "
        "max() instead of sum(). Whether two decodes on one Apple GPU actually "
        "overlap is the question this exists to answer",
        {"DHWANI_MIX_PARALLEL": "1"}),

    cfg("commit_min_1.5", "close windows sooner, so less of the clip is left for the "
        "end-of-clip decode", {"DHWANI_COMMIT_MIN_S": "1.5"}),

    # The temperature ladder re-decodes the same audio once per temperature when
    # whisper's thresholds fire — up to SIX sequential decodes, on a
    # speculation that _finalize joins for the whole remaining budget. Biggest
    # latency lever here that does not change a model, and genuinely
    # double-edged: the ladder is what stops the repetition loops that cap a
    # clip at 30. Both rows measured, neither assumed.
    cfg("temps_short", "a two-step temperature ladder instead of six",
        {"DHWANI_FINAL_TEMPS": "0.0,0.4"}),
    cfg("temps_off", "no ladder at all — fastest possible final, and the row most "
        "likely to reintroduce a repetition loop. Watch the flip/loop counts, "
        "not just the total", {"DHWANI_FINAL_TEMPS": "0.0"}),
]


# --- models and decoding ----------------------------------------------------
# All of these were measured for QUALITY on a slow box, in a harness that never
# ran the deadline. Their cost was therefore invisible. Re-run through the real
# streaming path they are being scored on speed and quality together, which is
# the only comparison that decides anything.

MODELS = [
    # Greedy is now the DEFAULT and the control. This row asks the opposite
    # question: does this Mac's mlx_whisper support beam search at all, and if
    # it does, is the +4.97/70 measured on CTranslate2 real here? It is the only
    # row allowed to switch beam on, and the ladder means a refusal costs one
    # call rather than the transcript. If the run log shows the beam rung being
    # rejected, that IS the answer builderr already gave us, confirmed locally.
    cfg("mlx_beam_5", "does beam search work on real mlx_whisper, and is it worth "
        "what it costs — the question two rounds died on without ever being asked",
        {"DHWANI_MLX_BEAM": "1", "DHWANI_BEAM": "5"}),
    cfg("mix_beam_5", "beam on the mix model: measured +0.56/70 for 2.7x the decode, "
        "rejected on a box that could not see what 2.7x costs",
        {"DHWANI_MIX_BEAM": "5"}),
    cfg("model_medium", "whisper medium as the primary — beaten on quality by turbo, "
        "but its 24-layer decoder is the thing that made finals run 2137-5271ms",
        {"DHWANI_MODEL": "medium"}),
    cfg("model_large_v3", "the full large-v3 (8-bit mlx). Slower per clip, and the "
        "English losses that cap clips at 50 are rare-word mishearings a bigger "
        "model is the only known fix for", {"DHWANI_MODEL": "large-v3"}),
    cfg("en_model_large_v3", "large-v3 for ENGLISH finals only. English never pays "
        "for the mix decode, so it has budget to spare — and every English loss "
        "on this corpus is a fact flip worth the whole 20-point axis",
        {"DHWANI_EN_MODEL": "large-v3"}),
    cfg("mix_off", "what the second model is worth once its time is counted",
        {"DHWANI_MIX_MODEL": ""}),
    # The mix decode is the whole end-to-final on every Indic clip, and it was
    # running on the slowest runtime available: transformers/MPS at 2.16s
    # against mlx's 1.08s for the same size class. Converting it is now the
    # default; this row forces the old path back to price the change.
    cfg("mix_transformers", "the OLD mix runtime — measures what converting the "
        "mix model to MLX actually bought",
        {"DHWANI_MIX_BACKEND": "transformers"}),
    cfg("mix_only_off", "run the primary on pure Hindi again. Worth +5.8/70 of "
        "quality and a whole extra decode; the default now says that trade is "
        "negative once the clock is counted", {"DHWANI_MIX_ONLY": "0"}),
    cfg("mix_apex", "the Apache-2.0 alternative, in case OpenRAIL is outside "
        "builderr's licensing bar. Needs the codeswitch gate: it measured 5.6/70 "
        "on pure Devanagari",
        {"DHWANI_MIX_MODEL": "Oriserve/Whisper-Hindi2Hinglish-Apex",
         "DHWANI_MIX_GATE": "codeswitch"}),
    cfg("mix_hindi_medium", "the Hindi specialist: +0.60/70 for +60% decode time, "
        "rejected partly because it is trained on FLEURS and FLEURS is most of "
        "this corpus. Kept in for completeness, and it must be read with that "
        "contamination in mind", {"DHWANI_MIX_MODEL": "vasista22/whisper-hindi-medium"}),
    cfg("draft_tiny", "a cheaper partial model, so the committer and the speculator "
        "wait behind less", {"DHWANI_DRAFT_MODEL": "tiny"}),
    cfg("backend_ct2", "CTranslate2 on CPU instead of MLX on the GPU. Assumed slower "
        "on Apple silicon and never actually checked on Apple silicon",
        {"DHWANI_BACKEND": "ctranslate2"}),
]


# Rows whose model is a multi-gigabyte download all of its own. Dropping them
# takes the run from roughly 20 GB of models to roughly 10 GB, and costs only
# the model-comparison arm — every latency finding survives. SKIP_BIG_MODELS=1.
BIG_MODEL_ROWS = {"model_large_v3", "mix_apex", "mix_hindi_medium", "model_medium"}

ALL = BASELINES + LATENCY + MODELS


def by_id(cid):
    for c in ALL:
        if c["id"] == cid:
            return c
    return None


def smoke_subset():
    """Five minutes, tiny models, one clip per category: proves the plumbing
    works before anyone commits three hours to it. SMOKE=1."""
    tiny = {"DHWANI_MODEL": "tiny", "DHWANI_DRAFT_MODEL": "tiny", "DHWANI_MIX_MODEL": ""}
    one = {"fleurs_english": 1, "fleurs_hindi": 1, "openslr104_hinglish": 1}
    rows = [cfg("smoke_stream", "does the fast instrument run at all", dict(tiny),
                runs=1, timeout_s=900),
            cfg("smoke_evaluator", "does the sealed server run at all",
                dict(tiny), mode="evaluator", runs=1, timeout_s=900)]
    for r in rows:
        r["slice"] = one
    return rows


def quick_subset():
    """~45 minutes instead of ~3 hours: the baseline calibration plus the rows
    most likely to move the number. Used when QUICK=1."""
    keep = {"baseline_evaluator", "baseline_stream_slice", "budget_3.0", "chunk_0",
            "partials_off", "spec_join_periodic_4", "mix_transformers", "mix_only_off",
            "en_model_large_v3", "temps_short"}
    out = [c for c in ALL if c["id"] in keep]
    for c in out:
        if c["mode"] == "evaluator":
            c["runs"], c["slice"] = 3, SWEEP_SLICE
    return out
