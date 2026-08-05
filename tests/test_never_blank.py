"""A blank final must be impossible. This is the suite that was missing.

builderr ran two submitted revisions and could not publish a score for either:
"the default MLX path requests beam search, which the scoring runtime does not
support, so every final came back blank." Two rounds, several weeks, zero.

Nothing in the old suite could have caught it, because every test faked
`_transcribe` — the level ABOVE the backend — so the backend call shape, which
is the thing that was wrong, was never exercised. These tests fake
`mlx_whisper` itself and assert the one property that actually matters: **text
comes out**, whatever the runtime refuses to accept.

Run:  python -m pytest tests/test_never_blank.py -q
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solution import draft as D
from tests.test_dhwani import speech          # noqa: E402


# --- a fake mlx_whisper whose API we control -------------------------------

def install_fake_mlx(monkeypatch, *, reject=(), result=None, calls=None):
    """Install a fake `mlx_whisper` that refuses any call mentioning a kwarg in
    `reject`, mimicking a runtime that does not support that option."""
    def transcribe(pcm, **kwargs):
        if calls is not None:
            calls.append(kwargs)
        for bad in reject:
            if bad in kwargs:
                raise TypeError(f"transcribe() got an unexpected keyword argument {bad!r}")
        return dict(result if result is not None else _WORDS_RESULT)

    module = types.ModuleType("mlx_whisper")
    module.transcribe = transcribe            # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)
    monkeypatch.setattr(D, "_backend", "mlx")
    D._MLX_LEVEL.clear()
    return module


_WORDS_RESULT = {
    "language": "en",
    "text": " hello world",
    "segments": [{"text": " hello world",
                  "words": [{"word": " hello", "start": 0.0, "end": 0.4},
                            {"word": " world", "start": 0.4, "end": 0.9}]}],
}
# What a build that ignores word_timestamps returns: real text, no word spans.
_NO_WORDS_RESULT = {
    "language": "en",
    "text": " hello world",
    "segments": [{"text": " hello world"}],
}
# And one that does not even populate segments.
_TEXT_ONLY_RESULT = {"language": "en", "text": " hello world", "segments": []}


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for name in ("DHWANI_BEAM", "DHWANI_MLX_BEAM", "DHWANI_LANG", "DHWANI_VERIFY"):
        monkeypatch.delenv(name, raising=False)
    D._MLX_LEVEL.clear()
    yield
    D._MLX_LEVEL.clear()
    D._backend = None


# --- the reported bug -------------------------------------------------------

def test_beam_is_greedy_by_default():
    """builderr asked for exactly this: 'please make greedy mode the default'."""
    assert D._beam_size(True) == 1
    assert D._beam_size(False) == 1


def test_mlx_never_requests_beam_search_by_default(monkeypatch):
    """The other half of their ask: 'or send a revision that does not require
    MLX beam search'. Not one call may mention it."""
    calls: list = []
    install_fake_mlx(monkeypatch, calls=calls)
    D._transcribe_mlx(speech(2.0), lang="en", prompt="p", model_name="tiny", final=True)
    assert calls, "no call was made at all"
    assert not any("beam_size" in kw for kw in calls), (
        f"beam search requested on the MLX path: {calls}")


def test_mlx_beam_even_when_opted_into_degrades_instead_of_blanking(monkeypatch):
    """DHWANI_MLX_BEAM is opt-in, and a runtime that refuses it must still
    return a transcript — the failure mode being fixed is blankness, not beam."""
    monkeypatch.setenv("DHWANI_MLX_BEAM", "1")
    monkeypatch.setenv("DHWANI_BEAM", "5")
    calls: list = []
    install_fake_mlx(monkeypatch, reject=("beam_size",), calls=calls)

    words, lang = D._transcribe_mlx(speech(2.0), lang="en", prompt="",
                                    model_name="tiny", final=True)
    assert any("beam_size" in kw for kw in calls), "never even tried the opt-in"
    assert D._text_from(words, lang).strip(), "a rejected beam blanked the decode"


def test_the_exact_reported_failure_beam_search_not_implemented(monkeypatch):
    """builderr's runtime, reproduced as reported. mlx_whisper accepts
    `beam_size` as a field and then refuses to USE it — so the exception is not
    a TypeError, and the old `except TypeError` guard never saw it. It went
    straight out of _transcribe_mlx, out of every decode in the process, and
    came back as a blank final on every clip of two submitted rounds."""
    monkeypatch.setenv("DHWANI_MLX_BEAM", "1")
    monkeypatch.setenv("DHWANI_BEAM", "5")

    def transcribe(pcm, **kwargs):
        if "beam_size" in kwargs:
            raise NotImplementedError("beam search decoder is not implemented")
        return dict(_WORDS_RESULT)

    module = types.ModuleType("mlx_whisper")
    module.transcribe = transcribe            # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)
    monkeypatch.setattr(D, "_backend", "mlx")
    D._MLX_LEVEL.clear()

    words, lang = D._transcribe_mlx(speech(2.0), lang="en", prompt="ctx",
                                    model_name="tiny", final=True)
    assert "hello world" in D._text_from(words, lang), (
        "the exact failure builderr reported still blanks the final")


def test_one_unsupported_kwarg_does_not_blank_the_decode(monkeypatch):
    """The exact shape of the shipped bug: the old fallback re-sent the kwargs
    that had just failed, from inside the except block that caught them, so a
    single bad keyword raised out of every decode in the process."""
    for bad in ("initial_prompt", "word_timestamps", "condition_on_previous_text",
                "logprob_threshold", "compression_ratio_threshold",
                "no_speech_threshold", "temperature", "task", "language"):
        D._MLX_LEVEL.clear()
        install_fake_mlx(monkeypatch, reject=(bad,))
        words, lang = D._transcribe_mlx(speech(2.0), lang="en", prompt="ctx",
                                        model_name="tiny", final=True)
        assert D._text_from(words, lang).strip(), f"{bad!r} blanked the decode"


def test_a_runtime_that_refuses_everything_but_the_bare_call_still_works(monkeypatch):
    install_fake_mlx(monkeypatch, reject=(
        "beam_size", "initial_prompt", "word_timestamps", "temperature",
        "condition_on_previous_text", "compression_ratio_threshold",
        "logprob_threshold", "no_speech_threshold", "language", "task"))
    words, lang = D._transcribe_mlx(speech(2.0), lang="en", prompt="ctx",
                                    model_name="tiny", final=True)
    assert D._text_from(words, lang).strip()


def test_a_runtime_that_refuses_everything_raises_rather_than_lying(monkeypatch):
    """If literally no call works, raise — so the caller's fallbacks run and the
    self-check can demote the backend. Silently returning '' is the one
    behaviour that must never happen."""
    def transcribe(pcm, **kwargs):
        raise RuntimeError("this runtime is broken")
    module = types.ModuleType("mlx_whisper")
    module.transcribe = transcribe            # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_whisper", module)
    monkeypatch.setattr(D, "_backend", "mlx")
    with pytest.raises(Exception):
        D._transcribe_mlx(speech(2.0), lang="en", prompt="", model_name="tiny", final=True)


# --- the silent one: no exception, no words, no score ----------------------

def test_result_without_word_spans_still_produces_the_transcript(monkeypatch):
    """No exception is raised here at all. The old code read segments[i]['words'],
    got nothing, and returned a BLANK final while result['text'] held the
    transcript the whole time. Nothing in a log would show it."""
    install_fake_mlx(monkeypatch, result=_NO_WORDS_RESULT)
    words, lang = D._transcribe_mlx(speech(2.0), lang="en", prompt="",
                                    model_name="tiny", final=True)
    assert "hello world" in D._text_from(words, lang), (
        "segments without word spans produced a blank final")


def test_result_with_only_top_level_text_still_produces_the_transcript(monkeypatch):
    install_fake_mlx(monkeypatch, result=_TEXT_ONLY_RESULT)
    words, lang = D._transcribe_mlx(speech(2.0), lang="en", prompt="",
                                    model_name="tiny", final=True)
    assert "hello world" in D._text_from(words, lang)


def test_a_genuinely_empty_result_stays_empty(monkeypatch):
    """The fallback must not invent text where the model produced none."""
    install_fake_mlx(monkeypatch, result={"language": "en", "text": "  ", "segments": []})
    words, _ = D._transcribe_mlx(speech(2.0), lang="en", prompt="",
                                 model_name="tiny", final=True)
    assert words == []


def test_ctranslate2_empty_word_spans_fall_back_to_segment_text(monkeypatch):
    """Identical hazard on the other backend, fixed the same way."""
    class Seg:
        text = " hello world"
        words = None

    class Model:
        def transcribe(self, audio, **kwargs):
            return iter([Seg()]), types.SimpleNamespace(language="en")

    monkeypatch.setattr(D, "_get_model_ctranslate2",
                        lambda name: (Model(), D.threading.Lock()))
    words, lang = D._transcribe_ctranslate2(
        np.zeros(16000, dtype=np.float32), "en", "", "tiny", final=True)
    assert "hello world" in D._text_from(words, lang)


# --- the ladder pays for itself once, not once per clip --------------------

def test_the_working_rung_is_remembered(monkeypatch):
    calls: list = []
    install_fake_mlx(monkeypatch, reject=("initial_prompt",), calls=calls)

    D._transcribe_mlx(speech(1.0), lang="en", prompt="ctx", model_name="tiny", final=True)
    first = len(calls)
    calls.clear()
    D._transcribe_mlx(speech(1.0), lang="en", prompt="ctx", model_name="tiny", final=True)

    assert first > 1, "expected the first call to walk down the ladder"
    assert len(calls) == 1, (
        f"re-probed the ladder on every clip ({len(calls)} calls) — that is "
        "wasted latency on the axis worth 30 points")


# --- the self-check that would have caught this before submitting ----------

def test_self_check_demotes_mlx_when_the_final_path_raises(monkeypatch):
    monkeypatch.setattr(D, "_backend", "mlx")
    monkeypatch.setattr(D, "_transcribe",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no beam here")))
    monkeypatch.setattr(D, "_get_model_ctranslate2", lambda name: (object(), D.threading.Lock()))

    D._verify_final_path()
    assert D._backend == "ctranslate2", (
        "a final path that cannot run was left selected — this is precisely the "
        "state two submitted rounds shipped in")


def test_self_check_leaves_a_working_mlx_alone(monkeypatch):
    monkeypatch.setattr(D, "_backend", "mlx")
    monkeypatch.setattr(D, "_transcribe", lambda *a, **k: ([(" ok", 0.0, 1.0)], "en"))
    D._verify_final_path()
    assert D._backend == "mlx"


def test_self_check_keeps_mlx_if_the_fallback_is_also_broken(monkeypatch):
    """Demoting to a backend that cannot load either would trade one failure for
    a worse one. Stay put and let the per-decode ladders try."""
    monkeypatch.setattr(D, "_backend", "mlx")
    monkeypatch.setattr(D, "_transcribe",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("broken")))
    monkeypatch.setattr(D, "_get_model_ctranslate2",
                        lambda name: (_ for _ in ()).throw(RuntimeError("also broken")))
    D._verify_final_path()
    assert D._backend == "mlx"


def test_self_check_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("DHWANI_VERIFY", "0")
    monkeypatch.setattr(D, "_backend", "mlx")
    monkeypatch.setattr(D, "_transcribe",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    D._verify_final_path()
    assert D._backend == "mlx"


# --- and the whole engine, end to end --------------------------------------

def test_finalize_never_returns_blank_while_any_decode_still_works(monkeypatch):
    """The integration guarantee. Every scored path fails; only the cheapest
    possible call succeeds. The clip must still score something, because a
    blank scores zero and a bad transcript does not."""
    monkeypatch.setenv("DHWANI_LANG", "hi")
    monkeypatch.setenv("DHWANI_MIX_MODEL", "")
    monkeypatch.setenv("DHWANI_PARTIALS", "0")
    monkeypatch.setenv("DHWANI_SPECULATE", "0")
    monkeypatch.setenv("DHWANI_HARD_WAIT_S", "1")

    def fake(window, lang, prompt, final=False, model=None, fast=False):
        if final:
            raise RuntimeError("the scored decode shape is unsupported here")
        return [(" salvaged", 0.0, 1.0)], "hi"

    monkeypatch.setattr(D, "_transcribe", fake)
    D.draft_reset()

    text, stable = D.draft(speech(3.0), True)
    assert text.strip(), "returned a BLANK final with a working decode available"
    assert "salvaged" in text
    assert stable == len(text)
    assert D._LAST_FINAL_PATH == "last-resort"


def test_finalize_reports_blank_honestly_when_nothing_works(monkeypatch):
    """If truly every decode is dead the final is blank — but it must be
    LABELLED blank, so the local harness reports a zero instead of a mystery."""
    monkeypatch.setenv("DHWANI_LANG", "hi")
    monkeypatch.setenv("DHWANI_MIX_MODEL", "")
    monkeypatch.setenv("DHWANI_PARTIALS", "0")
    monkeypatch.setenv("DHWANI_SPECULATE", "0")
    monkeypatch.setenv("DHWANI_HARD_WAIT_S", "1")
    monkeypatch.setattr(D, "_transcribe",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("everything is dead")))
    D.draft_reset()

    text, _ = D.draft(speech(2.0), True)
    assert text == ""
    assert D._LAST_FINAL_PATH == "blank"
