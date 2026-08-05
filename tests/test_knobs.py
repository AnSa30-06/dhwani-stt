"""Experiment knobs added for the Apple-hardware sweep (macbench/).

Every flag here defaults to the SHIPPED behaviour. These tests exist so the
sweep can turn each one on knowing it does what its name says, and — the part
that actually matters — that leaving it off changes nothing about the engine
that was submitted.

No model, no network: the same fake transcriber the main suite uses.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solution import draft as D
from tests.test_dhwani import (  # noqa: E402  (shared fakes, one definition)
    BPS,
    DECODE_COST_S,
    engine,           # noqa: F401  (pytest fixture, used by name)
    feed_partials,
    make_fake,
    make_two_model_fake,
    silence,
    speech,
    wait_for_speculation,
)


def _slow_devanagari(delay: float = 0.5):
    def fake(window, lang, prompt, final=False, model=None, fast=False):
        time.sleep(delay)
        return [(" नमस्ते", 0.0, 0.5), (" दुनिया", 0.5, 1.0)], "hi"
    return fake


def _wait_for_committer(timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while D._fc_alive() and time.monotonic() < deadline:
        time.sleep(0.02)


# --- the guard that matters most --------------------------------------------

def test_every_experiment_flag_defaults_to_shipped_behaviour(monkeypatch):
    """A sweep knob that silently altered the submitted engine when unset would
    make every measurement taken before it a lie."""
    for name in ("DHWANI_SPEC_JOIN", "DHWANI_SPEC_PERIODIC_S", "DHWANI_MIX_PARALLEL",
                 "DHWANI_PARTIALS", "DHWANI_FC_LANG_PIN", "DHWANI_SPEC_SILENCE_S",
                 "DHWANI_SPEC_MIN_AUDIO_S", "DHWANI_COMMIT_MIN_S", "DHWANI_PAUSE_S",
                 "DHWANI_SETTLE_S", "DHWANI_MIN_DECODE_S", "DHWANI_MIX_MIN_S",
                 "DHWANI_HARD_WAIT_S"):
        monkeypatch.delenv(name, raising=False)

    assert D._spec_join_enabled() is False
    assert D._env_f("DHWANI_SPEC_PERIODIC_S", 0.0) == 0.0
    assert D._env_flag("DHWANI_MIX_PARALLEL") is False
    assert D._partials_enabled() is True
    assert D._fc_lang_pin() is True
    assert D._env_f("DHWANI_SPEC_SILENCE_S", D.SPEC_SILENCE_S) == D.SPEC_SILENCE_S
    assert D._env_f("DHWANI_PAUSE_S", D.PAUSE_S) == D.PAUSE_S
    assert D._env_f("DHWANI_MIX_MIN_S", D.MIX_MIN_S) == D.MIX_MIN_S
    assert D._env_f("DHWANI_COMMIT_MIN_S", D.COMMIT_MIN_S) == D.COMMIT_MIN_S
    assert D._env_f("DHWANI_SETTLE_S", D.CHUNK_SETTLE_S) == D.CHUNK_SETTLE_S


def test_env_f_survives_junk_and_empty(monkeypatch):
    """These get written by a shell script on someone else's laptop; a typo
    must degrade to the default, never crash a three-hour run."""
    monkeypatch.setenv("DHWANI_PAUSE_S", "not-a-number")
    assert D._env_f("DHWANI_PAUSE_S", 0.35) == 0.35
    monkeypatch.setenv("DHWANI_PAUSE_S", "")
    assert D._env_f("DHWANI_PAUSE_S", 0.35) == 0.35
    monkeypatch.setenv("DHWANI_PAUSE_S", "0.5")
    assert D._env_f("DHWANI_PAUSE_S", 0.35) == 0.5


def test_env_flag_reads_the_usual_spellings(monkeypatch):
    for off in ("0", "false", "no", "off"):
        monkeypatch.setenv("DHWANI_X", off)
        assert D._env_flag("DHWANI_X", True) is False
    for on in ("1", "true", "yes"):
        monkeypatch.setenv("DHWANI_X", on)
        assert D._env_flag("DHWANI_X", False) is True


# --- DHWANI_PARTIALS --------------------------------------------------------

def test_partials_off_runs_no_unscored_decode(engine, monkeypatch):
    """Partials are worth zero points by the published protocol. With them off,
    the only decodes that may happen are finals — nothing else touches the
    accelerator the scored decode needs."""
    monkeypatch.setenv("DHWANI_PARTIALS", "0")
    monkeypatch.setenv("DHWANI_SPECULATE", "0")     # isolate the partial worker
    record: list = []
    monkeypatch.setattr(D, "_transcribe", make_fake(record))
    D.draft_reset()

    feed_partials(speech(3.0))
    time.sleep(0.2)
    assert not any(not r["final"] for r in record), (
        f"partial worker ran with DHWANI_PARTIALS=0: {record}")

    text, _ = D.draft(speech(3.0), True)
    assert text.strip(), "the final broke with partials disabled"


def test_partials_on_by_default_still_run(engine, monkeypatch):
    record: list = []
    monkeypatch.setenv("DHWANI_SPECULATE", "0")
    monkeypatch.setattr(D, "_transcribe", make_fake(record))
    D.draft_reset()
    feed_partials(speech(3.0))
    time.sleep(0.4)
    assert any(not r["final"] for r in record), "partial worker never ran by default"


# --- DHWANI_SPEC_PERIODIC_S -------------------------------------------------

def test_periodic_speculation_arms_without_any_silence(engine, monkeypatch):
    """A speaker who never pauses gets no speculation today, so their final is a
    cold whole-buffer decode. This is the fix being measured."""
    monkeypatch.setenv("DHWANI_SPEC_PERIODIC_S", "1.0")
    D.draft_reset()
    feed_partials(speech(4.0))       # speech right up to the last frame
    assert wait_for_speculation(), "periodic speculation never armed"


def test_periodic_off_means_silence_is_still_required(engine, monkeypatch):
    monkeypatch.delenv("DHWANI_SPEC_PERIODIC_S", raising=False)
    D.draft_reset()
    feed_partials(speech(4.0))
    time.sleep(0.4)
    with D._state_lock:
        assert D._spec_text is None, "speculated with no silence and no periodic knob"


# --- DHWANI_SPEC_JOIN -------------------------------------------------------

def test_spec_join_decodes_only_the_part_the_speculation_missed(engine, monkeypatch):
    """The point of the knob. A speculation overtaken by speech is still a
    full-quality decode of everything before that point, and today the whole
    thing is discarded and the buffer re-decoded from zero."""
    monkeypatch.setenv("DHWANI_SPEC_JOIN", "1")
    record: list = []
    monkeypatch.setattr(D, "_transcribe", make_fake(record))
    D.draft_reset()

    head = speech(2.0) + silence(1.0)
    audio = head + speech(2.0)          # speaker resumed, then the clip ends
    feed_partials(head)
    assert wait_for_speculation(), "no prefix speculation to join"
    feed_partials(audio)

    record.clear()
    text, _ = D.draft(audio, True)

    assert D._LAST_FINAL_PATH == "spec-join", (
        f"took the {D._LAST_FINAL_PATH!r} path instead of joining")
    assert text.strip()
    finals = [r for r in record if r["final"]]
    assert finals, "spec-join did no tail decode at all"
    assert max(r["nbytes"] for r in finals) < len(audio) * 0.8, (
        f"re-decoded the whole buffer instead of just the tail: {finals}")


def test_spec_join_off_by_default_takes_the_full_tail_decode(engine, monkeypatch):
    monkeypatch.delenv("DHWANI_SPEC_JOIN", raising=False)
    D.draft_reset()
    head = speech(2.0) + silence(1.0)
    audio = head + speech(2.0)
    feed_partials(head)
    wait_for_speculation()
    feed_partials(audio)

    D.draft(audio, True)
    assert D._LAST_FINAL_PATH != "spec-join", "joined with the knob unset"


def test_spec_join_stands_down_when_the_speculation_covers_everything(engine, monkeypatch):
    """A complete speculation is free; joining would only add a decode nobody
    needs. _spec_take must still win."""
    monkeypatch.setenv("DHWANI_SPEC_JOIN", "1")
    D.draft_reset()
    audio = speech(3.0) + silence(1.0)
    feed_partials(audio)
    assert wait_for_speculation()

    t0 = time.monotonic()
    D.draft(audio, True)
    assert D._LAST_FINAL_PATH == "speculation", (
        f"joined instead of taking the complete speculation: {D._LAST_FINAL_PATH!r}")
    assert time.monotonic() - t0 < DECODE_COST_S / 2


def test_spec_join_stands_down_when_committed_windows_reach_further(engine, monkeypatch):
    """If the committer already locked past the speculation, _final_decode's own
    tail is the shorter one — joining would be strictly worse."""
    monkeypatch.setenv("DHWANI_SPEC_JOIN", "1")
    D.draft_reset()
    with D._state_lock:
        D._spec_text, D._spec_covered = "prefix", 2 * BPS
        D._fc_text, D._fc_bytes = "committed", 3 * BPS
    assert D._start_spec_join(speech(5.0), None) is None


def test_spec_join_stands_down_with_nothing_speculated(engine, monkeypatch):
    monkeypatch.setenv("DHWANI_SPEC_JOIN", "1")
    D.draft_reset()
    assert D._start_spec_join(speech(5.0), None) is None


# --- DHWANI_FC_LANG_PIN -----------------------------------------------------

def test_fc_lang_pin_off_redetects_on_every_window(engine, monkeypatch):
    """The named suspect in the DHWANI_CHUNK_S=6 regression, isolated from the
    window size that exposed it."""
    monkeypatch.setenv("DHWANI_FC_LANG_PIN", "0")
    D.draft_reset()
    with D._state_lock:
        D._fc_lang = "hi"           # as if an earlier window had pinned Hindi
    seen: list = []
    monkeypatch.setattr(D, "_decode_final_segment",
                        lambda audio, pinned_lang, prompt="", fast=False, deadline=None:
                        (seen.append(pinned_lang) or ("text", "hi")))
    D._maybe_commit_window(speech(20.0))
    _wait_for_committer()
    assert seen == [None], f"window inherited a pinned language: {seen}"


def test_fc_lang_pin_on_by_default_reuses_the_pin(engine, monkeypatch):
    monkeypatch.delenv("DHWANI_FC_LANG_PIN", raising=False)
    D.draft_reset()
    with D._state_lock:
        D._fc_lang = "hi"
    seen: list = []
    monkeypatch.setattr(D, "_decode_final_segment",
                        lambda audio, pinned_lang, prompt="", fast=False, deadline=None:
                        (seen.append(pinned_lang) or ("text", "hi")))
    D._maybe_commit_window(speech(20.0))
    _wait_for_committer()
    assert seen == ["hi"], f"dropped the pin with the knob unset: {seen}"


# --- DHWANI_MIX_PARALLEL ----------------------------------------------------

def test_mix_parallel_overlaps_the_two_decodes_on_pure_hindi(monkeypatch):
    """Pure Hindi is the slowest final this engine produces, because it is the
    one case that pays for BOTH models in series. Overlapped it should cost
    about one decode, not two."""
    monkeypatch.setenv("DHWANI_MIX_PARALLEL", "1")
    monkeypatch.delenv("DHWANI_LANG", raising=False)
    monkeypatch.setenv("DHWANI_MIX_MODEL", "mixy-hinglish")
    monkeypatch.setenv("DHWANI_MIX_BACKEND", "native")
    monkeypatch.setattr(D, "_transcribe", _slow_devanagari(0.5))
    D.draft_reset()
    monkeypatch.setattr(D, "_lang", "hi")

    t0 = time.monotonic()
    text, _ = D._decode_final_segment(speech(3.0), pinned_lang=None,
                                      deadline=time.monotonic() + 30.0)
    elapsed = time.monotonic() - t0

    assert text.strip()
    assert elapsed < 0.9, f"decodes did not overlap: {elapsed:.2f}s for two 0.5s decodes"


def test_mix_parallel_off_by_default_stays_sequential(monkeypatch):
    monkeypatch.delenv("DHWANI_MIX_PARALLEL", raising=False)
    monkeypatch.delenv("DHWANI_LANG", raising=False)
    monkeypatch.setenv("DHWANI_MIX_ONLY", "0")   # mix-only would skip the second decode
    monkeypatch.setenv("DHWANI_MIX_MODEL", "mixy-hinglish")
    monkeypatch.setenv("DHWANI_MIX_BACKEND", "native")
    monkeypatch.setattr(D, "_transcribe", _slow_devanagari(0.5))
    D.draft_reset()
    monkeypatch.setattr(D, "_lang", "hi")

    t0 = time.monotonic()
    D._decode_final_segment(speech(3.0), pinned_lang=None,
                            deadline=time.monotonic() + 30.0)
    assert time.monotonic() - t0 >= 0.9, "ran in parallel with the knob unset"


def test_mix_parallel_still_skips_the_primary_on_code_switched_audio(monkeypatch):
    """The measured mix-first win must survive the knob: on Hinglish the mix
    answer returns and the primary thread is simply abandoned unread."""
    monkeypatch.setenv("DHWANI_MIX_PARALLEL", "1")
    monkeypatch.delenv("DHWANI_LANG", raising=False)
    monkeypatch.setenv("DHWANI_MIX_MODEL", "mixy-hinglish")
    monkeypatch.setenv("DHWANI_MIX_BACKEND", "native")
    record: list = []
    monkeypatch.setattr(D, "_transcribe", make_two_model_fake(record))
    D.draft_reset()
    monkeypatch.setattr(D, "_lang", "hi")

    text, lang = D._decode_final_segment(speech(3.0), pinned_lang=None)
    assert lang == "hi"
    assert "hello" in text, f"lost the mix model's transcript: {text!r}"


def test_mix_parallel_falls_back_to_mix_when_the_primary_overruns(monkeypatch):
    """A parallel primary that misses the deadline must degrade to the mix
    answer, never to a blank final — blank is the hardest cap on the card."""
    monkeypatch.setenv("DHWANI_MIX_PARALLEL", "1")
    monkeypatch.delenv("DHWANI_LANG", raising=False)
    monkeypatch.setenv("DHWANI_MIX_MODEL", "mixy-hinglish")
    monkeypatch.setenv("DHWANI_MIX_BACKEND", "native")

    def fake(window, lang, prompt, final=False, model=None, fast=False):
        if model is None:            # the primary: far too slow to fit
            time.sleep(5.0)
        return [(" नमस्ते", 0.0, 0.5), (" दुनिया", 0.5, 1.0)], "hi"

    monkeypatch.setattr(D, "_transcribe", fake)
    D.draft_reset()
    monkeypatch.setattr(D, "_lang", "hi")

    text, _ = D._decode_final_segment(speech(3.0), pinned_lang=None,
                                      deadline=time.monotonic() + 1.0)
    assert text.strip(), "a parallel primary overrun produced a BLANK final"


# --- the swept timing constants actually reach the code ---------------------

def test_pause_s_knob_changes_where_a_window_closes(engine, monkeypatch):
    """_pause_end is the function the commit-timing knobs exist to move; if the
    env never reached it the whole sweep axis would be inert."""
    audio = speech(4.0) + silence(0.5) + speech(4.0) + silence(1.5)
    min_bytes = int(2.5 * BPS)

    monkeypatch.setenv("DHWANI_PAUSE_S", "0.2")
    lenient = D._pause_end(audio, 0, min_bytes)
    monkeypatch.setenv("DHWANI_PAUSE_S", "5.0")     # no pause this long exists
    strict = D._pause_end(audio, 0, min_bytes)

    assert lenient is not None, "no boundary found even with a 0.2s pause"
    assert strict is None, "found a 5s pause in a clip that has none"


def test_spec_silence_knob_changes_when_a_speculation_arms(engine, monkeypatch):
    rms = D._frame_rms(speech(2.0) + silence(0.2))
    thr = D._silence_threshold(rms)

    monkeypatch.setenv("DHWANI_SPEC_SILENCE_S", "0.1")
    assert D._tail_is_silent(rms, thr), "0.2s of silence did not satisfy a 0.1s window"
    monkeypatch.setenv("DHWANI_SPEC_SILENCE_S", "1.0")
    assert not D._tail_is_silent(rms, thr), "0.2s of silence satisfied a 1.0s window"
