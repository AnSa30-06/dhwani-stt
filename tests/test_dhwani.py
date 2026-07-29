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
    def fake(window, lang, prompt, final=False, model=None, fast=False):
        if record is not None:
            record.append({"nbytes": len(window), "lang": lang, "final": final,
                           "model": model, "fast": fast})
        if final:
            time.sleep(DECODE_COST_S)
        secs = max(1, round(len(window) / BPS))
        words = [(f" w{i}", float(i), float(i) + 0.5) for i in range(secs)]
        return words, raw_lang
    return fake


@pytest.fixture()
def engine(monkeypatch):
    """Fresh engine state, fake transcriber, forced language (no detect calls).
    The mix model is disabled ("" — it defaults ON) so no real model loads."""
    monkeypatch.setenv("DHWANI_LANG", "hi")
    monkeypatch.setenv("DHWANI_MIX_MODEL", "")
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
    # The fake emits one word per second of window, so the full 4s clip yields
    # 4 words and the stale 3s speculation only 3. Assert COVERAGE rather than
    # equality with a re-decode: a committer may legitimately have locked part
    # of the clip by now, so re-running _final_decode is a moving target.
    assert len(text.split()) == 4, f"final did not cover the resumed speech: {text!r}"


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


def test_router_uses_transformers_backend_for_hf_repo(monkeypatch):
    """A mix model named like an HF repo (contains '/') routes through the
    transformers path, not the native backend."""
    calls: list = []
    monkeypatch.delenv("DHWANI_LANG", raising=False)
    monkeypatch.setenv("DHWANI_MIX_MODEL", "someone/some-hinglish-model")
    monkeypatch.setattr(D, "_transcribe", make_fake(raw_lang="hi"))
    monkeypatch.setattr(
        D, "_transcribe_mix_transformers",
        lambda window: (calls.append(len(window)) or [(" mix out", 0.0, 2.0)], "hi"))
    D.draft_reset()

    text = D._final_decode(speech(2.0))
    assert calls, "transformers mix path was never invoked"
    assert "mix" in text or text.strip(), f"unexpected final: {text!r}"


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


def make_fake_devanagari(record=None):
    """Fake whose transcript is pure Devanagari — no code-switch signal."""
    def fake(window, lang, prompt, final=False, model=None, fast=False):
        if record is not None:
            record.append({"model": model, "final": final, "fast": fast})
        secs = max(1, round(len(window) / BPS))
        return [(f" शब्द{i}", float(i), float(i) + 0.5) for i in range(secs)], "hi"
    return fake


def test_default_mix_model_is_zero_stt(monkeypatch):
    monkeypatch.delenv("DHWANI_MIX_MODEL", raising=False)
    assert D._mix_model() == "shunyalabs/zero-stt-hinglish"
    monkeypatch.setenv("DHWANI_MIX_MODEL", "")
    assert D._mix_model() is None


def test_default_gate_escalates_pure_hindi(monkeypatch):
    """Default gate is 'indic': even a pure-Devanagari primary escalates —
    measured safe because zero-stt ties the default model on pure Hindi."""
    monkeypatch.delenv("DHWANI_LANG", raising=False)
    monkeypatch.setenv("DHWANI_MIX_MODEL", "someone/mix-model")
    monkeypatch.delenv("DHWANI_MIX_GATE", raising=False)
    monkeypatch.setattr(D, "_transcribe", make_fake_devanagari())
    called = []
    monkeypatch.setattr(
        D, "_transcribe_mix_transformers",
        lambda window: (called.append(1) or [(" शब्द0 शब्द1 extra", 0.0, 2.0)], "hi"))
    D.draft_reset()
    D._final_decode(speech(2.0))
    assert called, "default gate should escalate Hindi-detected clips"


def test_codeswitch_gate_keeps_pure_hindi_off_mix_model(monkeypatch):
    """DHWANI_MIX_GATE=codeswitch: a pure-Devanagari primary must NOT escalate
    (the Apex-style protection; Apex measured 5.6/70 on pure Hindi)."""
    monkeypatch.delenv("DHWANI_LANG", raising=False)
    monkeypatch.setenv("DHWANI_MIX_MODEL", "someone/mix-model")
    monkeypatch.setenv("DHWANI_MIX_GATE", "codeswitch")
    monkeypatch.setattr(D, "_transcribe", make_fake_devanagari())
    called = []
    monkeypatch.setattr(D, "_transcribe_mix_transformers",
                        lambda window: (called.append(1) or [(" x", 0.0, 1.0)], "hi"))
    D.draft_reset()

    text = D._final_decode(speech(2.0))
    assert not called, "pure-Devanagari clip escalated despite codeswitch gate"
    assert "शब्द" in text


def test_mix_candidate_rejected_when_it_drops_a_number(monkeypatch):
    """Measured on Apex: '334' came back as '3.3 0.4'. If the healthy primary
    heard a multi-digit number the mix candidate lost, keep the primary."""
    monkeypatch.delenv("DHWANI_LANG", raising=False)
    monkeypatch.setenv("DHWANI_MIX_MODEL", "someone/mix-model")

    def primary(window, lang, prompt, final=False, model=None, fast=False):
        return ([(" version", 0.0, 0.5), (" 334", 0.5, 1.0), (" use", 1.0, 1.5),
                 (" करो", 1.5, 2.0)], "hi")
    monkeypatch.setattr(D, "_transcribe", primary)
    monkeypatch.setattr(
        D, "_transcribe_mix_transformers",
        lambda window: ([(" version teen point teen use karo bahut hi badhiya"
                          " tarika hai yeh", 0.0, 2.0)], "hi"))
    D.draft_reset()

    text = D._final_decode(speech(2.0))
    assert "334" in text, f"number was lost to the mix candidate: {text!r}"


def test_drops_numbers_helper():
    assert D._drops_numbers("version 334 use karo", "version teen use karo")
    assert not D._drops_numbers("version 334 use karo", "version 334 hi to hai")
    assert not D._drops_numbers("koi number nahi", "still no number")
    # single digits are ignored
    assert not D._drops_numbers("le lo 5 cheezen", "le lo paanch cheezen")
    # the guard sees number-normalized text: 3.3.4 == 334
    assert not D._drops_numbers("version 334", "version 3.3.4")


# --- chunked final (bounded end-of-clip latency on long clips) -------------

_MARK_OFFSET = 4000   # keeps amplitude well above the silence floor while still
                      # letting a window's samples name the seconds they cover


def marked(seconds: int) -> bytes:
    """PCM where every sample in second k holds a loud, distinct value, so a
    decoder can read a window's byte-slice and name exactly which seconds it spans."""
    n = seconds * D.SR
    return (np.arange(n) // D.SR + 1 + _MARK_OFFSET).astype(np.int16).tobytes()


def marked_fake(record: list | None = None, cost_per_s: float = 0.02,
                raw_lang: str = "en"):
    def fake(window, lang, prompt, final=False, model=None, fast=False):
        pcm = np.frombuffer(window, dtype=np.int16)
        if final and cost_per_s:
            time.sleep(len(window) / BPS * cost_per_s)   # decode cost ~ length
        vals = [v for v in np.unique(pcm) if v > _MARK_OFFSET]   # skip silence (zeros)
        words = [(f" s{int(v) - _MARK_OFFSET}", float(i), float(i) + 0.4)
                 for i, v in enumerate(vals)]
        if record is not None:
            record.append({"nbytes": len(window), "final": final, "nwords": len(words)})
        return words, raw_lang
    return fake


def drain_committers(audio: bytes, timeout_s: float = 15.0) -> None:
    """Keep ticking draft() until every closeable window has been committed and
    no committer is in flight (mimics the server's steady partial cadence)."""
    deadline = time.monotonic() + timeout_s
    stable, last = 0, -1
    while time.monotonic() < deadline:
        D.draft(audio, False)
        with D._state_lock:
            fb, busy = D._fc_bytes, D._fc_busy
        if not busy and fb == last:
            stable += 1
            if stable >= 3:
                return
        else:
            stable, last = 0, fb
        time.sleep(0.05)


@pytest.fixture()
def chunk_engine(monkeypatch):
    monkeypatch.delenv("DHWANI_LANG", raising=False)
    monkeypatch.setenv("DHWANI_MIX_MODEL", "")
    monkeypatch.setenv("DHWANI_SPECULATE", "0")   # isolate chunking from speculation
    monkeypatch.setattr(D, "_transcribe", marked_fake())
    D.draft_reset()
    yield D
    D.draft_reset()


def test_committer_runs_and_leaves_a_bounded_tail(chunk_engine):
    audio = marked(90)
    feed_partials(audio)
    drain_committers(audio)
    with D._state_lock:
        fc_bytes = D._fc_bytes
    assert 0 < fc_bytes < len(audio), "committer never closed a window, or ate the whole clip"
    tail_s = (len(audio) - fc_bytes) / BPS
    assert tail_s <= D._chunk_s() + D.CHUNK_SETTLE_S + 2.0, f"tail is {tail_s:.1f}s, not bounded"


def test_final_latency_is_bounded_on_long_clips(chunk_engine):
    audio = marked(90)
    feed_partials(audio)
    drain_committers(audio)
    t0 = time.monotonic()
    text, _ = D.draft(audio, True)
    dt = time.monotonic() - t0
    # a whole-buffer decode of 90s would cost ~90*0.02 = 1.8s; the bounded tail
    # is ~one window (~24s -> ~0.5s). Give headroom for a possible in-flight window.
    assert dt < 1.1, f"final took {dt*1000:.0f}ms — tail was not bounded"
    assert text.strip()


def test_chunked_final_much_faster_than_whole_buffer(monkeypatch):
    monkeypatch.delenv("DHWANI_LANG", raising=False)
    monkeypatch.setenv("DHWANI_MIX_MODEL", "")
    monkeypatch.setenv("DHWANI_SPECULATE", "0")
    monkeypatch.setattr(D, "_transcribe", marked_fake())
    audio = marked(90)

    monkeypatch.setenv("DHWANI_CHUNK_S", "0")     # whole-buffer baseline
    D.draft_reset()
    feed_partials(audio)
    t0 = time.monotonic(); D.draft(audio, True); whole = time.monotonic() - t0

    monkeypatch.setenv("DHWANI_CHUNK_S", "24")    # chunked
    D.draft_reset()
    feed_partials(audio)
    drain_committers(audio)
    t0 = time.monotonic(); D.draft(audio, True); chunked = time.monotonic() - t0

    assert chunked < whole * 0.6, f"chunked {chunked*1000:.0f}ms vs whole {whole*1000:.0f}ms"


def test_chunked_final_keeps_the_whole_transcript(chunk_engine):
    audio = marked(90)
    feed_partials(audio)
    drain_committers(audio)
    text, _ = D.draft(audio, True)
    toks = set(text.split())
    assert "s1" in toks and "s90" in toks, "final dropped the start or the end of the clip"
    assert len({t for t in toks if t.startswith("s")}) >= 72, "final lost too many seconds"


def test_short_clip_is_not_chunked(chunk_engine):
    audio = marked(5)               # well under one 24s window
    feed_partials(audio)
    drain_committers(audio)
    with D._state_lock:
        assert D._fc_bytes == 0, "a short clip should never close a window"
    text, _ = D.draft(audio, True)
    assert "s1" in text and "s5" in text


def test_chunk_disabled_env_never_commits(monkeypatch):
    monkeypatch.delenv("DHWANI_LANG", raising=False)
    monkeypatch.setenv("DHWANI_MIX_MODEL", "")
    monkeypatch.setenv("DHWANI_CHUNK_S", "0")
    monkeypatch.setattr(D, "_transcribe", marked_fake())
    D.draft_reset()
    audio = marked(90)
    feed_partials(audio)
    drain_committers(audio)
    with D._state_lock:
        assert D._fc_bytes == 0, "chunking should be off with DHWANI_CHUNK_S=0"


def test_language_pinned_from_first_window(monkeypatch):
    monkeypatch.delenv("DHWANI_LANG", raising=False)
    monkeypatch.setenv("DHWANI_MIX_MODEL", "")
    monkeypatch.setenv("DHWANI_SPECULATE", "0")
    monkeypatch.setattr(D, "_transcribe", marked_fake(raw_lang="hi"))
    D.draft_reset()
    audio = marked(60)
    feed_partials(audio)
    drain_committers(audio)
    with D._state_lock:
        assert D._fc_lang == "hi", "language was not pinned from the first closed window"


def slow_fake(seconds: float, raw_lang: str = "en"):
    """A decoder far slower than any budget — stands in for the M1 running the
    temperature ladder or the transformers mix model on an awkward clip."""
    def fake(window, lang, prompt, final=False, model=None, fast=False):
        time.sleep(seconds)
        return [(" slow", 0.0, 1.0)], raw_lang
    return fake


def prime_partials(audio: bytes, timeout_s: float = 3.0) -> str:
    """Run the fast partial path until it has produced text to fall back on."""
    feed_partials(audio)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with D._state_lock:
            got = (D._committed + D._tail).strip()
        if got:
            return got
        D.draft(audio, False)
        time.sleep(0.05)
    return ""


def test_final_never_exceeds_budget_when_text_is_in_hand(monkeypatch):
    """THE guarantee: with a usable transcript already in hand, the final must
    return within budget no matter how slow the model is — a late final scores
    zero on latency and caps the clip."""
    monkeypatch.setenv("DHWANI_LANG", "hi")
    monkeypatch.setenv("DHWANI_MIX_MODEL", "")
    monkeypatch.setenv("DHWANI_SPECULATE", "0")
    monkeypatch.setenv("DHWANI_CHUNK_S", "0")
    monkeypatch.setattr(D, "_transcribe", make_fake())
    D.draft_reset()
    audio = speech(6.0)
    assert prime_partials(audio), "partials produced nothing; test cannot run"

    monkeypatch.setenv("DHWANI_FINAL_BUDGET_S", "0.5")
    monkeypatch.setattr(D, "_transcribe", slow_fake(10.0))
    t0 = time.monotonic()
    text, stable = D.draft(audio, True)
    dt = time.monotonic() - t0

    assert dt < 2.0, f"final blocked {dt:.1f}s on a 10s decoder — budget not enforced"
    assert text.strip() and stable == len(text)


def test_waits_past_budget_rather_than_return_a_blank_final(monkeypatch):
    """A blank final scores 0 for the clip; a late one still scores its quality
    (capped). So with nothing in hand, the budget must be overridden."""
    monkeypatch.setenv("DHWANI_LANG", "hi")
    monkeypatch.setenv("DHWANI_MIX_MODEL", "")
    monkeypatch.setenv("DHWANI_SPECULATE", "0")
    monkeypatch.setenv("DHWANI_CHUNK_S", "0")
    monkeypatch.setenv("DHWANI_FINAL_BUDGET_S", "0.2")
    monkeypatch.setattr(D, "_transcribe", slow_fake(1.0))   # slower than budget
    D.draft_reset()

    t0 = time.monotonic()
    text, _ = D.draft(speech(3.0), True)     # no partials primed: nothing in hand
    dt = time.monotonic() - t0

    assert text.strip(), "returned a blank final instead of waiting"
    assert dt > 0.2, "did not actually wait past the budget"


def test_budget_overrun_still_returns_text_not_blank(monkeypatch):
    """When the budget is missed we fall back to text already in hand. A blank
    final scores 0, so the fallback must carry whatever the partials produced."""
    monkeypatch.setenv("DHWANI_LANG", "hi")
    monkeypatch.setenv("DHWANI_MIX_MODEL", "")
    monkeypatch.setenv("DHWANI_SPECULATE", "0")
    monkeypatch.setenv("DHWANI_CHUNK_S", "0")
    monkeypatch.setattr(D, "_transcribe", make_fake())   # fast partials
    D.draft_reset()

    audio = speech(4.0)
    feed_partials(audio)                     # partial worker commits real text
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        with D._state_lock:
            if (D._committed + D._tail).strip():
                break
        D.draft(audio, False)
        time.sleep(0.05)

    monkeypatch.setenv("DHWANI_FINAL_BUDGET_S", "0.2")
    monkeypatch.setattr(D, "_transcribe", slow_fake(10.0))
    text, _ = D.draft(audio, True)
    assert text.strip(), "budget overrun produced a blank final"


def test_draft_reset_clears_the_busy_latch(monkeypatch):
    """Regression: a partial worker still in flight when a clip ended used to
    leave _busy latched, silencing partials on every later clip — and those
    partials are the fallback the final returns when a decode overruns."""
    monkeypatch.setenv("DHWANI_LANG", "hi")
    monkeypatch.setenv("DHWANI_MIX_MODEL", "")
    monkeypatch.setenv("DHWANI_SPECULATE", "0")
    monkeypatch.setattr(D, "_transcribe", slow_fake(5.0))
    D.draft_reset()
    D.draft(speech(3.0), False)                 # spawns a worker that will hang
    time.sleep(0.1)
    assert D._busy, "no worker started, test cannot prove the latch"

    D.draft_reset()                              # next clip begins
    assert not D._busy, "_busy stayed latched across draft_reset()"

    monkeypatch.setattr(D, "_transcribe", make_fake())
    audio = speech(3.0)
    feed_partials(audio)
    deadline = time.monotonic() + 3
    got = ""
    while time.monotonic() < deadline:
        with D._state_lock:
            got = (D._committed + D._tail).strip()
        if got:
            break
        D.draft(audio, False)
        time.sleep(0.05)
    assert got, "partials never ran on the clip after a hung worker"


def test_tail_decode_uses_the_fast_path(monkeypatch):
    """The end-of-clip decode must ask for fast=True (no temperature ladder),
    while a speculative/committed decode asks for full quality."""
    record: list = []
    monkeypatch.setenv("DHWANI_LANG", "hi")
    monkeypatch.setenv("DHWANI_MIX_MODEL", "")
    monkeypatch.setenv("DHWANI_SPECULATE", "0")
    monkeypatch.setattr(D, "_transcribe", make_fake(record))
    D.draft_reset()

    D.draft(speech(3.0), True)
    finals = [r for r in record if r["final"]]
    assert finals and all(r["fast"] for r in finals), (
        f"final decode did not use the fast path: {finals}")

    record.clear()
    D.draft_reset()
    D._final_decode(speech(3.0), fast=False)     # what speculation/commits run
    assert any(not r["fast"] for r in record), "quality path lost its ladder"


def test_fast_path_still_runs_the_mix_model(monkeypatch):
    """The mix model must run even on the fast path. It writes English terms in
    LATIN, and the scorecard greps must_have terms as Latin substrings — with
    turbo alone the Hinglish clips wrote इंप्रेस/टिटोरिल for impress/tutorial
    and lost the whole facts axis. `fast` drops the temperature ladder, not
    this."""
    monkeypatch.delenv("DHWANI_LANG", raising=False)
    monkeypatch.setenv("DHWANI_MIX_MODEL", "someone/mix-model")
    monkeypatch.setattr(D, "_transcribe", make_fake_devanagari())
    called: list = []
    monkeypatch.setattr(D, "_transcribe_mix_transformers",
                        lambda window: (called.append(1) or [(" mix", 0.0, 1.0)], "hi"))
    D.draft_reset()

    D._final_decode(speech(3.0), fast=True)
    assert called, "fast path dropped the mix model — that is the 4-clip bug"


def test_mix_model_skipped_only_when_the_deadline_is_spent(monkeypatch):
    """Quality when affordable, speed when not: with no time left the extra
    decode is skipped rather than blowing the budget."""
    monkeypatch.delenv("DHWANI_LANG", raising=False)
    monkeypatch.setenv("DHWANI_MIX_MODEL", "someone/mix-model")
    monkeypatch.setattr(D, "_transcribe", make_fake_devanagari())
    called: list = []
    monkeypatch.setattr(D, "_transcribe_mix_transformers",
                        lambda window: (called.append(1) or [(" mix", 0.0, 1.0)], "hi"))

    D.draft_reset()
    D._final_decode(speech(3.0), fast=True, deadline=time.monotonic() - 1)
    assert not called, "ran the mix decode with the deadline already blown"

    D.draft_reset()
    D._final_decode(speech(3.0), fast=True, deadline=time.monotonic() + 30)
    assert called, "skipped the mix decode despite ample time"


def test_spoken_numbers_become_digits():
    """The scorer requires each gold number verbatim; a spoken "सौ साल" against
    a gold "100 साल" is a fact flip. Measured on fleurs_hi_in_test_1718."""
    n = D._normalize_numbers
    assert "100" in n("कुछ कर एजंसिया सो साल से")
    assert "25" in n("twenty five to thirty") and "30" in n("twenty five to thirty")
    assert "334" in n("three hundred thirty four")
    assert "2020" in n("do hazaar bees")
    # conservative: a lone small number word is far more often an article
    assert "एक" in n("एक प्रस्तुति document")
    assert "one" in n("I have one idea")
    # already-digit text is left alone
    assert n("version 334 ka upyog") == "version 334 ka upyog"


def test_commits_at_a_pause_keep_the_tail_short(monkeypatch):
    """With a pause after COMMIT_MIN_S, a window should close there rather than
    waiting for the full CHUNK_S — that is what shrinks the end-of-clip tail."""
    monkeypatch.delenv("DHWANI_LANG", raising=False)
    monkeypatch.setenv("DHWANI_MIX_MODEL", "")
    monkeypatch.setenv("DHWANI_SPECULATE", "0")
    monkeypatch.setenv("DHWANI_CHUNK_S", "24")     # far longer than the clip
    monkeypatch.setattr(D, "_transcribe", marked_fake())
    D.draft_reset()

    # speech, a clear pause, then more speech — the pause is the boundary
    audio = marked(6) + silence(0.8) + marked(6) + silence(1.2)
    feed_partials(audio)
    drain_committers(audio)

    with D._state_lock:
        fc_bytes = D._fc_bytes
    assert fc_bytes > 0, "no window closed at the pause (tail stays long)"
    assert fc_bytes < len(audio), "committer consumed the whole clip"


def test_speculation_and_chunking_together_stay_instant(monkeypatch):
    monkeypatch.delenv("DHWANI_LANG", raising=False)
    monkeypatch.setenv("DHWANI_MIX_MODEL", "")
    monkeypatch.delenv("DHWANI_SPECULATE", raising=False)   # speculation ON
    monkeypatch.setattr(D, "_transcribe", marked_fake())
    D.draft_reset()
    audio = marked(60) + silence(1.0)
    feed_partials(audio)
    drain_committers(audio)
    # partials keep arriving through the trailing silence; those ticks (now that
    # the committer is idle) are what arm the speculative tail decode.
    for _ in range(6):
        D.draft(audio, False)
        if wait_for_speculation(0.1):
            break
    assert wait_for_speculation(4.0), "speculation never armed on a long clip"
    t0 = time.monotonic()
    text, _ = D.draft(audio, True)
    dt = time.monotonic() - t0
    assert dt < 0.3, f"final took {dt*1000:.0f}ms despite a ready speculation"
    assert "s60" in text
