"""dhwani solution tests: speculative finalization + language router.

No model, no network — a fake transcriber with a real decode cost stands in for
whisper. Run:  python -m pytest tests/test_dhwani.py -q
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solution import draft as D

BPS = D.BYTES_PER_SEC
DECODE_COST_S = 0.30    # what the fake charges per final decode, regardless of length


def speech(seconds: float) -> bytes:
    """Deterministic loud PCM: a 300-amplitude sawtooth scaled to ~3000 peak."""
    n = int(seconds * D.SR)
    wave = ((np.arange(n) % 20) - 10).astype(np.int16) * 300
    return wave.tobytes()


def silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(seconds * D.SR)


def make_fake(record: list | None = None, raw_lang: str = "hi"):
    def fake(window, lang, prompt, final=False, model=None):
        if record is not None:
            record.append({"nbytes": len(window), "lang": lang,
                           "final": final, "model": model})
        if final:
            time.sleep(DECODE_COST_S)
        secs = max(1, round(len(window) / BPS))
        words = [(f" w{i}", float(i), float(i) + 0.5) for i in range(secs)]
        return words, raw_lang
    return fake


@pytest.fixture()
def engine(monkeypatch):
    """Fresh engine state, fake transcriber, forced language (no detect calls)."""
    monkeypatch.setenv("DHWANI_LANG", "hi")
    monkeypatch.delenv("DHWANI_MIX_MODEL", raising=False)
    monkeypatch.delenv("DHWANI_SPECULATE", raising=False)
    monkeypatch.setattr(D, "_transcribe", make_fake())
    D.draft_reset()
    yield D
    D.draft_reset()


def feed_partials(audio: bytes, step_s: float = 0.5) -> None:
    """Mimic the sealed server: draft(False) on the growing buffer every ~500ms."""
    step = int(step_s * BPS)
    for end in range(step, len(audio) + 1, step):
        D.draft(audio[:end], False)


def wait_for_speculation(timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with D._state_lock:
            if D._spec_text is not None:
                return True
        time.sleep(0.02)
    return False


def test_speculative_final_is_instant_and_identical(engine):
    audio = speech(3.0) + silence(1.0)
    feed_partials(audio)
    assert wait_for_speculation(), "speculation never armed despite trailing silence"

    t0 = time.monotonic()
    text, stable = D.draft(audio, True)
    elapsed = time.monotonic() - t0

    assert elapsed < DECODE_COST_S / 2, (
        f"final took {elapsed * 1000:.0f}ms — speculation was not used")
    assert text.strip(), "speculative final is blank"
    assert stable == len(text)
    # byte-for-byte what a fresh decode of the same buffer produces
    assert text == D._final_decode(audio)


def test_abrupt_end_falls_back_to_fresh_decode(engine):
    audio = speech(3.0)     # speech to the last frame; nothing to speculate in
    feed_partials(audio)
    with D._state_lock:
        assert D._spec_text is None, "speculated with no trailing silence"

    t0 = time.monotonic()
    text, _ = D.draft(audio, True)
    elapsed = time.monotonic() - t0

    assert elapsed >= DECODE_COST_S * 0.8, "no fresh decode happened"
    assert text == D._final_decode(audio)


def test_resumed_speech_invalidates_speculation(engine):
    head = speech(2.0) + silence(1.0)
    audio = head + speech(1.0)          # speaker resumed, clip ends abruptly
    feed_partials(head)
    wait_for_speculation()
    feed_partials(audio)                 # the resumed speech arrives

    text, _ = D.draft(audio, True)
    # the fake emits one word per second of window: the full 4s clip decodes to
    # 4 words, the stale 3s speculation to 3 — the final must reflect all 4s.
    assert text == D._final_decode(audio)
    assert len(text.split()) == 4


def test_speculation_never_arms_on_pure_silence(engine):
    audio = silence(3.0)
    feed_partials(audio)
    with D._state_lock:
        assert D._spec_text is None


def test_draft_reset_discards_speculation(engine):
    audio = speech(2.0) + silence(1.0)
    feed_partials(audio)
    wait_for_speculation()
    D.draft_reset()
    with D._state_lock:
        assert D._spec_text is None and D._spec_covered == 0


def test_speculate_env_kill_switch(engine, monkeypatch):
    monkeypatch.setenv("DHWANI_SPECULATE", "0")
    audio = speech(2.0) + silence(1.0)
    feed_partials(audio)
    time.sleep(0.1)
    with D._state_lock:
        assert D._spec_text is None


def test_silence_detection_is_gain_invariant(engine):
    audio = speech(2.0) + silence(1.0)
    quiet = (np.frombuffer(audio, dtype=np.int16) * 0.1).astype(np.int16).tobytes()
    for buf in (audio, quiet):
        rms = D._frame_rms(buf)
        thr = D._silence_threshold(rms)
        assert D._tail_is_silent(rms, thr), "trailing silence not seen at this gain"
        assert D._speech_after(rms, thr, 0), "speech not seen at this gain"


def test_router_escalates_hindi_to_mix_model(monkeypatch):
    record: list = []
    monkeypatch.delenv("DHWANI_LANG", raising=False)
    monkeypatch.setenv("DHWANI_MIX_MODEL", "mixy-hinglish")
    monkeypatch.setattr(D, "_transcribe", make_fake(record, raw_lang="hi"))
    D.draft_reset()

    text = D._final_decode(speech(2.0))
    assert text.strip()
    finals = [r for r in record if r["final"]]
    assert any(r["model"] == "mixy-hinglish" and r["lang"] == "hi" for r in finals), (
        f"mix model never used: {finals}")


def test_router_leaves_english_on_default_model(monkeypatch):
    record: list = []
    monkeypatch.delenv("DHWANI_LANG", raising=False)
    monkeypatch.setenv("DHWANI_MIX_MODEL", "mixy-hinglish")
    monkeypatch.setattr(D, "_transcribe", make_fake(record, raw_lang="en"))
    D.draft_reset()

    D._final_decode(speech(2.0))
    finals = [r for r in record if r["final"]]
    assert all(r["model"] is None for r in finals), (
        f"English clip escalated to the mix model: {finals}")
