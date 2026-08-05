"""The budget policy, after it was measured and reversed.

For three rounds the final ran under a 3.0s budget on the reasoning that the
latency curve pays nothing past 5000ms. That read the curve and ignored the
caps. streaming_scorecard applies its caps to the TOTAL, and they are gentle:

    past 4000ms  -> clip capped at 70
    past 6000ms  -> clip capped at 50
    dropped      -> clip capped at 0      (evaluator gives up at 20s)

Measured on one clip, one machine, two runs minutes apart, differing only in
which side of the budget the decode landed on:

    2512ms  real decode     meaning 0.86, no flip     -> 83.2
    3005ms  fallback text   meaning 0.17, FACT FLIP   -> 24.7

and the counterfactual, had it been allowed to finish: 75.2 at 3500ms, 69.2 at
4500ms, 63.2 at 5500ms, 50.0 even at 6500ms. Every one beats 24.7.

So the policy is now: never trade quality for the clock. Be fast by finishing
sooner, never by returning worse text on time.
"""
from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solution import draft as D
from streaming_scorecard import end_to_final_points  # noqa: E402
from tests.test_dhwani import feed_partials, make_fake, silence, slow_fake, speech  # noqa: E402


# --- the arithmetic that drove the change ----------------------------------

def test_a_late_real_decode_beats_a_punctual_fragment():
    """This is the whole argument, expressed against the real scorecard rather
    than against anyone's intuition. If a future change makes the budget tight
    again, this is the test that should stop it."""
    real_meaning, fragment_meaning = 0.864, 0.174

    def clip_score(meaning, ms, flipped):
        base = 50.0 * meaning + (0.0 if flipped else 20.0) + end_to_final_points(ms)
        caps = [50.0] if flipped else []
        if ms > 6000:
            caps.append(50.0)
        elif ms > 4000:
            caps.append(70.0)
        return min([base, *caps]) if caps else base

    punctual_fragment = clip_score(fragment_meaning, 3005, flipped=True)
    assert round(punctual_fragment, 1) == 24.7        # what actually happened

    for late_ms in (3500, 4500, 5500, 6500):
        late_real = clip_score(real_meaning, late_ms, flipped=False)
        assert late_real > punctual_fragment, (
            f"a real decode at {late_ms}ms scores {late_real:.1f}, which should "
            f"beat the {punctual_fragment:.1f} the fallback scored")


def test_default_budget_sits_below_the_6000ms_cap_edge():
    """5.4s leaves the clip capped at 70 rather than 50, with margin for the
    websocket hop the harness measures on top of our own clock."""
    for name in ("DHWANI_FINAL_BUDGET_S",):
        os.environ.pop(name, None)
    assert 4.0 < D._final_budget_s() < 6.0


# --- never exceed the harness's own patience -------------------------------

def test_total_wait_cannot_reach_the_harness_drop_timeout(monkeypatch):
    """evaluator.py waits 20s for a final and then marks the clip DROPPED,
    which caps it at zero — worse than any late transcript. The budget plus the
    never-return-blank wait used to add up to 23s."""
    monkeypatch.delenv("DHWANI_HARD_STOP_S", raising=False)
    assert D._env_f("DHWANI_HARD_STOP_S", 18.0) < 20.0


def test_a_hung_decode_returns_before_the_harness_gives_up(monkeypatch):
    """End to end: nothing in hand, a decoder that never finishes, and the call
    must still come back inside the harness's 20s window."""
    monkeypatch.setenv("DHWANI_LANG", "hi")
    monkeypatch.setenv("DHWANI_MIX_MODEL", "")
    monkeypatch.setenv("DHWANI_SPECULATE", "0")
    monkeypatch.setenv("DHWANI_CHUNK_S", "0")
    monkeypatch.setenv("DHWANI_PARTIALS", "0")
    monkeypatch.setenv("DHWANI_FINAL_BUDGET_S", "0.4")
    monkeypatch.setenv("DHWANI_HARD_STOP_S", "1.2")
    monkeypatch.setattr(D, "_transcribe", slow_fake(30.0))
    D.draft_reset()

    t0 = time.monotonic()
    D.draft(speech(2.0), True)
    dt = time.monotonic() - t0
    assert dt < 4.0, f"blocked {dt:.1f}s — the harness would have dropped the clip"


# --- the coverage gate ------------------------------------------------------

def test_fallback_is_trusted_when_a_real_decode_covers_the_clip(monkeypatch):
    monkeypatch.delenv("DHWANI_BE_MIN_COVER", raising=False)
    D.draft_reset()
    audio = speech(5.0)
    with D._state_lock:
        D._fc_text, D._fc_bytes = "covered", len(audio)
    assert D._best_effort_is_trustworthy(audio)


def test_fallback_is_refused_when_only_partials_exist(monkeypatch):
    """Coverage zero means the only text is the rolling 1-3s LocalAgreement
    window on the small draft model — the text that scored 0.17."""
    monkeypatch.delenv("DHWANI_BE_MIN_COVER", raising=False)
    D.draft_reset()
    audio = speech(5.0)
    with D._state_lock:
        D._committed, D._tail = "some ", "partial words"
    assert not D._best_effort_is_trustworthy(audio)


def test_a_completed_speculation_also_counts_as_coverage(monkeypatch):
    """A speculation is a whole-buffer decode with full context — exactly as
    trustworthy as a committed window, and it reports its own coverage."""
    monkeypatch.delenv("DHWANI_BE_MIN_COVER", raising=False)
    D.draft_reset()
    audio = speech(5.0)
    with D._state_lock:
        D._spec_text, D._spec_covered = "speculated text", len(audio)
    assert D._best_effort_is_trustworthy(audio)


def test_coverage_threshold_is_tunable(monkeypatch):
    D.draft_reset()
    audio = speech(10.0)
    with D._state_lock:
        D._fc_text, D._fc_bytes = "half", len(audio) // 2
    monkeypatch.setenv("DHWANI_BE_MIN_COVER", "0.9")
    assert not D._best_effort_is_trustworthy(audio)
    monkeypatch.setenv("DHWANI_BE_MIN_COVER", "0.3")
    assert D._best_effort_is_trustworthy(audio)


# --- mix on MLX -------------------------------------------------------------

def test_mix_mlx_is_never_converted_on_the_scored_path(monkeypatch):
    """Conversion downloads a checkpoint and rewrites it. That belongs in
    warm_models(), while the network is up and nothing is timed — never inside
    a final, where it would be both a network call and a 30-second stall."""
    monkeypatch.setenv("DHWANI_MIX_MODEL", "someone/mix-model")
    monkeypatch.setenv("DHWANI_MLX_CACHE", os.path.join(os.path.dirname(__file__),
                                                        "_no_such_cache"))
    called: list = []
    monkeypatch.setattr(D, "_convert_mix_to_mlx",
                        lambda repo, out: called.append(repo))
    assert D._mix_mlx_dir(convert=False) is None
    assert not called, "a decode-time call tried to convert the mix model"


def test_mix_decode_falls_back_to_transformers_without_a_conversion(monkeypatch):
    monkeypatch.setenv("DHWANI_MIX_MODEL", "someone/mix-model")
    monkeypatch.setenv("DHWANI_MIX_BACKEND", "mlx")
    monkeypatch.setattr(D, "_mix_mlx_dir", lambda convert=False: None)
    used: list = []
    monkeypatch.setattr(D, "_transcribe_mix_transformers",
                        lambda window: (used.append(1) or [(" mix out", 0.0, 1.0)], "hi"))
    assert D._mix_decode(speech(2.0)).strip()
    assert used, "no conversion available and the transformers path did not run"


def test_mix_decode_uses_the_converted_model_when_present(monkeypatch):
    monkeypatch.setenv("DHWANI_MIX_MODEL", "someone/mix-model")
    monkeypatch.setenv("DHWANI_MIX_BACKEND", "mlx")
    monkeypatch.setattr(D, "_mix_mlx_dir", lambda convert=False: "/tmp/converted")
    seen: list = []

    def fake(window, lang, prompt, final=False, model=None, fast=False):
        seen.append(model)
        return [(" mixed", 0.0, 1.0)], "hi"

    monkeypatch.setattr(D, "_transcribe", fake)
    assert D._mix_decode(speech(2.0)).strip()
    assert seen == ["/tmp/converted"], f"did not use the converted model: {seen}"


def test_mix_mlx_is_OFF_by_default(monkeypatch):
    """Measured on real Apple silicon: the converted weights decode without the
    fine-tune's suppress-token recipe and ramble — 2230ms/57.7q became
    4835ms/55.7q on the same clip. Off unless someone opts in AND the real-clip
    verify blesses it."""
    monkeypatch.delenv("DHWANI_MIX_MLX", raising=False)
    monkeypatch.setenv("DHWANI_MIX_MODEL", "someone/mix-model")
    assert D._mix_mlx_dir(convert=True) is None


def test_mix_mlx_verify_fails_closed_without_a_reference_clip(monkeypatch, tmp_path):
    """No proof, no marker: the fallback it would displace is known-good, so an
    unprovable optimisation must stay off rather than ship on faith — the
    first verify shipped on faith and blessed a degraded model."""
    monkeypatch.setattr(D.os.path, "exists",
                        lambda p: False if p.endswith(".wav") else os.path.exists(p))
    assert D._verify_mix_mlx(str(tmp_path)) is None
    assert not (tmp_path / "VERIFIED").exists()


def test_mix_mlx_verify_rejects_a_rambling_decode(monkeypatch, tmp_path):
    """The exact observed failure: Devanagari present, but looping."""
    wav = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "dev", "audio", "fleurs_hi_in_test_1666.wav")
    if not os.path.exists(wav):
        pytest.skip("reference clip not present")
    monkeypatch.setattr(D, "_transcribe_mlx",
                        lambda *a, **k: ([(" नमस्ते तो तो तो तो तो तो तो तो", 0.0, 1.0)], "hi"))
    assert D._verify_mix_mlx(str(tmp_path)) is None
    assert not (tmp_path / "VERIFIED").exists()


def test_mix_mlx_verify_blesses_a_clean_decode(monkeypatch, tmp_path):
    wav = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "dev", "audio", "fleurs_hi_in_test_1666.wav")
    if not os.path.exists(wav):
        pytest.skip("reference clip not present")
    monkeypatch.setattr(D, "_transcribe_mlx",
                        lambda *a, **k: ([(" नमस्ते दुनिया यह ठीक है", 0.0, 1.0)], "hi"))
    assert D._verify_mix_mlx(str(tmp_path)) == str(tmp_path)
    assert (tmp_path / "VERIFIED").exists()
