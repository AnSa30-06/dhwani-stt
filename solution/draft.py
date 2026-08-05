"""dhwani — streaming Hindi+English dictation for the builderr STT challenge.

Drop this in as `solution/draft.py`. It is self-contained: `orthography.py` is
optional and the engine degrades to a no-op adapter if it is absent.

The one thing that decides the 30-point latency axis: the sealed harness calls
`draft()` *synchronously* inside its WebSocket handler, so end-to-final latency
is exactly how long `draft(audio, is_final=True)` blocks. The scored final is a
fresh whole-buffer decode (see _final_decode for why streaming commits were
banished from it) — so the latency win comes from STARTING that decode early:
when the tail of the buffer goes silent for SPEC_SILENCE_S, a background thread
speculatively runs the exact same whole-buffer final decode during the silence.
If `end` arrives and no new speech landed after the speculative decode began,
the final returns in milliseconds; if speech resumed, the speculation is
discarded and the normal decode runs (never a wrong answer, only a slower one).
≤1000 ms end-to-final is the full 30 points on the published latency curve.

A second worker still decodes the rolling window for live partials
(LocalAgreement-2, Machacek et al.) — those are unscored UI hints only.

Two decode backends, chosen automatically (DHWANI_BACKEND=auto|mlx|ctranslate2):

  * mlx      — mlx_whisper on Apple's GPU/ANE via the MLX framework. This is
               the backend the frozen scoring box (M1 Pro) needs, because
               CTranslate2 (below) has no Metal path and runs CPU-only even
               there. UNVERIFIED: written from the published API shape and
               memory of the library's source, with no Apple Silicon machine
               to run it on. Run tools/mac_smoke_test.py before trusting it.
  * ctranslate2 — faster-whisper. CPU-only on Mac; this is the path measured
               and verified in README.md (mean quality 41.3/70 at `medium`).
               Used automatically when mlx_whisper isn't importable (i.e.
               anywhere that isn't Apple Silicon), so this file keeps working
               on Windows/Linux for continued testing.

Environment:
    DHWANI_BACKEND       auto | mlx | ctranslate2          (default: auto)
    DHWANI_MODEL         model size/repo for the final      (default: large-v3-turbo)
    DHWANI_DRAFT_MODEL   cheaper size/repo for partials     (default: small)
    DHWANI_BEAM          beam width for the scored final (default: 1, GREEDY).
                         Beam 5 measured +4.97/70 on the English clips — on
                         CTranslate2. Shipped to a scoring host that runs MLX,
                         it produced a BLANK final on every clip of two rounds.
                         See _beam_size. Greedy is now the default everywhere.
    DHWANI_MLX_BEAM      1 to let the MLX path ask for beam search at all
                         (default: off). Even then it is only the top rung of
                         the capability ladder, so a runtime that refuses it
                         loses one call, not the transcript.
    DHWANI_VERIFY        0 to skip the start-up self-check that runs the real
                         scored decode once and falls back from MLX to
                         CTranslate2 if it raises (default: on). This check is
                         the thing whose absence cost two submissions.
    DHWANI_EN_MODEL      optional stronger checkpoint for ENGLISH finals; they
                         never pay for the mix decode, so they have budget to
                         spare (unset = use DHWANI_MODEL)
    DHWANI_MIX_MODEL     Hindi/code-switch model for the final's Hindi path
                         (default: shunyalabs/zero-stt-hinglish, measured best
                         on Hinglish and tied with turbo on pure Hindi).
                         "" disables; any HF whisper fine-tune id works.
    DHWANI_MIX_GATE      indic (default: escalate every Hindi-detected clip)
                         | codeswitch (require Latin letters in the primary)
    DHWANI_MIX_BACKEND   transformers | native (default: transformers when the
                         model id contains "/")
    DHWANI_SPECULATE     0 to disable speculative finalization (default: ON)
    DHWANI_CHUNK_S       seconds; windows are committed during the clip (at
                         phrase boundaries, or at this hard bound if the speaker
                         never pauses) so the end-of-clip decode only processes
                         the uncommitted tail, not the whole buffer (default: 12
                         — 6 is faster but measured -6.90/70 once windows
                         actually close, see _chunk_s. 0 restores whole-buffer)
    DHWANI_FINAL_BUDGET_S  wall-clock the final may spend before falling back to
                         text already in hand (default: 5.4 — RAISED from 3.0
                         after measurement. The scorecard caps a late clip at 70
                         past 4s and 50 past 6s, while the fallback measured
                         24.7 against the real decode's 83.2 on the same clip.
                         See _final_budget_s for the full arithmetic)
    DHWANI_BE_MIN_COVER  fraction of the clip a real decode must already cover
                         before the fallback text is trusted (default: 0.6).
                         Below it the only text is the rolling partials, which
                         is what scored 0.17 meaning with a fact flip
    DHWANI_HARD_STOP_S   absolute ceiling on one final call (default: 18.0). The
                         harness DROPS a clip that has not answered in 20s and
                         scores it zero, which is worse than any late transcript
    DHWANI_MIX_ONLY      1 (default) returns the mix model's transcript on Indic
                         clips without also running the primary. The primary is
                         worth +5.8/70 on pure Hindi and costs a whole decode;
                         priced on real hardware that trade is negative. 0
                         restores the pair
    DHWANI_MIX_MLX       0 (default) — the mix model stays on transformers.
                         MEASURED OFF on real Apple silicon: the converted
                         weights are correct (encoder sinusoids intact, all 946
                         tensors mapped) but the fine-tune's decoding recipe
                         lives in generation_config.json — 88 suppress_tokens,
                         begin_suppress, forced_decoder_ids — and mlx_whisper
                         ignores all of it, so the model RAMBLES: same Hindi
                         clip 2230ms/57.7q via transformers vs 4835ms/55.7q via
                         the conversion. Slower AND worse. 1 re-enables the
                         experiment; the warm-time verify then requires a clean
                         non-looping decode of a real Hindi clip before use
    DHWANI_WARM_ON_IMPORT  unset (default) auto-detects and warms only inside the
                         sealed stream_server process, on a background thread.
                         The server prints READY before any model is loaded, so
                         without this the load lands on the FIRST scored clip.
                         1 forces it on, 0 off (the test suite and tooling).
    DHWANI_DEVICE        cpu | cuda | auto (ctranslate2 only)
    DHWANI_ORTHOGRAPHY   0 to disable the corpus adapter   (default: ON — the
                         must_have terms are Latin substrings, so Devanagari
                         renderings of them forfeit the 20-point facts axis)

Experiment knobs — every one of these DEFAULTS TO TODAY'S BEHAVIOUR, so an unset
environment runs the shipped engine unchanged. They exist because none of them
could be decided on the box this was written on, which decodes 10-20x slower
than the scoring host and therefore cannot read the 30-point latency axis at
all. macbench/ sweeps them on Apple hardware; whatever wins there gets promoted
to a default in code, with its number in the comment, the same as every other
constant here.

    DHWANI_SPEC_JOIN     1 = when speech overtook a speculation, decode only the
                         part it missed and join, instead of discarding a
                         full-quality decode of the whole prefix (default: 0)
    DHWANI_SPEC_PERIODIC_S  seconds; arm a speculation this often even without
                         trailing silence, so a speaker who never pauses still
                         gets one (default: 0 = silence-triggered only)
    DHWANI_MIX_PARALLEL  1 = start the primary decode alongside the mix decode
                         on Indic clips, making pure Hindi cost max() instead of
                         sum() (default: 0 = sequential mix-first)
    DHWANI_PARTIALS      0 = never run the unscored LocalAgreement partial
                         worker, freeing the accelerator entirely for the
                         decodes that do earn points (default: 1)
    DHWANI_FC_LANG_PIN   0 = re-detect the language on every committed window
                         instead of pinning it from the first (default: 1)
    DHWANI_SPEC_SILENCE_S / DHWANI_SPEC_MIN_AUDIO_S / DHWANI_COMMIT_MIN_S /
    DHWANI_PAUSE_S / DHWANI_SETTLE_S / DHWANI_MIN_DECODE_S /
    DHWANI_MIX_MIN_S / DHWANI_HARD_WAIT_S
                         the timing constants at the top of this file, made
                         sweepable; each defaults to its literal
"""
from __future__ import annotations

import os
import re
import threading

SR = 16000
BYTES_PER_SEC = SR * 2

MIN_DECODE_S = 0.8       # decoding a shorter window than this is mostly noise
LANG_DETECT_S = 3.0      # commit nothing until this much audio has picked a language
COMMIT_LAG_S = 0.3       # never commit a word that touches the live edge
FORCE_COMMIT_S = 6.0     # if agreement stalls, commit anyway so the tail stays short
PROMPT_CHARS = 160

SPEC_SILENCE_S = 0.3     # trailing silence that arms a speculative final decode
SPEC_MIN_AUDIO_S = 1.0   # never speculate on less audio than this
FRAME_S = 0.02           # harness frame size; silence detection works per frame

CHUNK_SETTLE_S = 1.0     # audio that must arrive past a window before it closes
COMMIT_MIN_S = 2.5       # smallest window worth closing at a pause. Lower =
                         # more of the clip is already decoded when `end`
                         # arrives, so the final decode faces a shorter tail
PAUSE_S = 0.35           # silence this long is a phrase boundary we can cut on
HARD_WAIT_S = 20.0       # only ever waited when the alternative is a blank final
                         # (blank scores 0; late still scores quality, capped)

_INDIC = {"hi", "mr", "ne", "sa", "ur", "bh", "mai"}
_PUNCT = re.compile(r"[^\w]", re.UNICODE)


def _env_f(name: str, default: float) -> float:
    """A tuning constant, overridable per-run.

    Every timing constant above was chosen on a box that decodes 10-20x slower
    than the scoring host, so none of them has ever been swept where it matters.
    Reading them through here costs one dict lookup per call and makes the whole
    set a sweepable axis on a machine that can actually measure the clock. The
    literal above stays the default, so an unset environment is byte-identical
    behaviour to before.
    """
    try:
        raw = os.environ.get(name)
        return default if raw in (None, "") else float(raw)
    except ValueError:
        return default


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw not in ("0", "false", "no", "off")

_models: dict[str, object] = {}
_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()
_backend: str | None = None

_state_lock = threading.Lock()
_committed = ""          # text we promised never to rewrite
_committed_bytes = 0     # audio fully accounted for by _committed
_tail = ""               # uncommitted hypothesis after _committed
_prev_words: list[tuple[str, float, float]] = []
_lang: str | None = None
_busy = False
_finalizing = False

# speculative-final state. _clip_gen guards against a stale speculation thread
# from a previous clip writing its result into the next one: draft_reset() bumps
# the generation and the thread only stores a result if its generation matches.
_clip_gen = 0
_spec_thread: threading.Thread | None = None
_spec_started = 0        # len(audio) when the in-flight speculation began
_spec_text: str | None = None
_spec_covered = 0        # len(audio) a COMPLETED speculation decoded

# Which branch _finalize took, for the local harness only. The scored path never
# reads it; it exists because quality alone cannot tell a real decode apart from
# the best-effort fallback, and those two differ by ~9 points on the same audio.
_LAST_FINAL_PATH = ""

# committed-window final state (see the "chunked final" section). On long clips
# closed windows are decoded during the clip and their text locked here, so the
# end-of-clip decode only has to handle the last, bounded window.
_fc_thread: threading.Thread | None = None
_fc_busy = False
_fc_text = ""            # locked transcript of all closed windows
_fc_bytes = 0            # audio fully accounted for by _fc_text
_fc_lang: str | None = None   # language pinned once, on the first closed window

try:
    from solution.orthography import map_words
except Exception:
    try:
        from orthography import map_words
    except Exception:
        def map_words(words, lang):
            return words


def _log_error(context: str, exc: BaseException) -> None:
    """The real harness only reads the (text, stable_chars) tuple draft() hands
    back, never stderr — so printing here is free diagnostic signal, not a
    protocol risk. Without this, a real failure (bad repo id, network hiccup,
    an mlx_whisper API mismatch) and "the model legitimately produced nothing"
    are indistinguishable: both come back as a blank final. They should not be."""
    import sys
    import traceback
    print(f"[dhwani] ERROR in {context}: {type(exc).__name__}: {exc}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)


def _log_note(message: str) -> None:
    """Expected degradation, not a failure. A capability ladder walking down a
    rung is normal traffic on an unfamiliar runtime; printing a traceback for
    each one buries the line that matters in the run log."""
    import sys
    print(f"[dhwani] {message}", file=sys.stderr)


# --- contract -------------------------------------------------------------

def draft_reset() -> None:
    global _committed, _committed_bytes, _tail, _prev_words, _lang, _finalizing
    global _clip_gen, _spec_thread, _spec_started, _spec_text, _spec_covered
    global _fc_thread, _fc_busy, _fc_text, _fc_bytes, _fc_lang, _busy
    with _state_lock:
        # _busy MUST be cleared here. A partial worker still in flight when the
        # clip ends would otherwise latch it into the next clip and silence
        # every partial — and those partials are the fallback text the final
        # returns when a decode overruns its budget. The bumped _clip_gen below
        # stops that stale worker writing into the new clip's state.
        _busy = False
        _committed = ""
        _committed_bytes = 0
        _tail = ""
        _prev_words = []
        _lang = None
        _finalizing = False
        _clip_gen += 1       # orphans any speculation/committer thread still running
        _spec_thread = None
        _spec_started = 0
        _spec_text = None
        _spec_covered = 0
        _fc_thread = None
        _fc_busy = False
        _fc_text = ""
        _fc_bytes = 0
        _fc_lang = None


def draft(audio_buffer: bytes, is_final: bool) -> tuple[str, int]:
    global _busy, _finalizing
    try:
        if is_final:
            with _state_lock:
                _finalizing = True
            return _finalize(audio_buffer)

        # On long clips, close and lock finished windows in the background so the
        # end-of-clip decode only faces the last window (see "chunked final").
        try:
            _maybe_commit_window(audio_buffer)
        except Exception as exc:
            _log_error("window committer (final falls back to a whole-buffer decode)", exc)

        # The LocalAgreement partial worker drives the live preview and is
        # unscored. It yields to the committer and the speculator so the scored
        # final decode never contends with it for the single accelerator.
        # (Letting it run alongside them was tried and measured worse: two
        # concurrent decodes on one accelerator slow both, and the partial is
        # the one that doesn't earn points.)
        if _partials_enabled() and not _busy and not _spec_alive() and not _fc_busy and \
                len(audio_buffer) - _committed_bytes >= \
                int(_env_f("DHWANI_MIN_DECODE_S", MIN_DECODE_S) * BYTES_PER_SEC):
            _busy = True
            try:
                threading.Thread(target=_worker, args=(audio_buffer, _clip_gen),
                                 daemon=True).start()
            except Exception:
                _busy = False   # a latched _busy would silence every later partial

        try:
            _maybe_speculate(audio_buffer)
        except Exception as exc:
            _log_error("speculation trigger (final falls back to a fresh decode)", exc)

        with _state_lock:
            return (_committed + _tail, len(_committed))
    except Exception as exc:
        _log_error("draft()", exc)
        with _state_lock:
            return (_committed + _tail, len(_committed))


# --- decoding -------------------------------------------------------------

def _partials_enabled() -> bool:
    """Whether to run the LocalAgreement partial worker at all.

    Partials are explicitly worth ZERO points — the protocol calls them optional
    messages that never affect the score — yet they are the only thing besides
    the speculator and the committer competing for the one accelerator. The
    worker already yields to both, but yielding is not the same as being absent:
    a partial that started a frame before the speculator wanted to arm still
    holds the GPU for its whole decode, and _maybe_speculate() refuses to arm
    while `_busy`. Turning them off is therefore a pure-latency experiment with
    no quality axis to lose, and it has never been measurable on a box where the
    clock itself could not be read.

    The cost, if any, is the `live` half of _best_effort_text(): with partials
    off, the fallback of last resort is the committed windows and a completed
    speculation only. Default ON — this is an experiment, not a decision.
    """
    return _env_flag("DHWANI_PARTIALS", True)


def _worker(audio: bytes, gen: int | None = None) -> None:
    global _busy
    try:
        _decode_and_commit(audio, final=False, gen=gen)
    except Exception as exc:
        _log_error("streaming worker (unscored, but logged for visibility)", exc)
    finally:
        if gen is None or gen == _clip_gen:
            _busy = False   # a newer clip already cleared it; don't stomp


def _finalize(audio: bytes) -> tuple[str, int]:
    """Produce the scored final under a hard time budget.

    Latency is 30 of the 100 points and a late final scores zero on that axis
    (and caps the clip), so the one thing this must never do is block for an
    unbounded decode. Order of preference, each strictly cheaper than the last:

      1. a completed speculative decode (full quality, already paid for during
         the speaker's trailing silence) — free;
      2. a FAST decode of just the uncommitted tail (greedy, no temperature
         ladder, no mix model) under whatever budget remains;
      3. whatever text we already hold (committed windows + the live partial),
         returned immediately.

    Quality is only sacrificed in (3), and only on clips where a decode could
    not finish in time — where the alternative was a late final worth nothing.
    """
    global _LAST_FINAL_PATH
    import time as _time
    deadline = _time.monotonic() + _final_budget_s()
    # The harness gives up on a final after 20s (evaluator.py: asyncio.wait_for
    # on the receive task) and scores the clip a hard ZERO for a dropped final —
    # worse than any late transcript, and not recoverable by quality. Every wait
    # below is bounded by this, so the sum of the budget and the
    # never-return-blank wait can never reach it. Previously they added to 23s.
    hard_stop = _time.monotonic() + _env_f("DHWANI_HARD_STOP_S", 18.0)

    # Joins the in-flight speculation for the WHOLE remaining budget, leaving
    # the tail decode nothing if that wait fails. Holding back a reserve for the
    # tail decode looks obviously safer and was measured WORSE: 75.37 -> 65.09
    # on the streaming harness at the scoring host's decode rate, with one clip
    # going 94.1 -> 50.9 because it abandoned a speculation that would have
    # landed. The speculation is a whole-buffer decode at full quality; the tail
    # decode replacing it is degraded AND contends with the abandoned
    # speculation, which keeps running. Waiting is the better bet.
    text = _spec_take(audio, deadline)
    if text is not None and text.strip():
        _LAST_FINAL_PATH = "speculation"
        return (text, len(text))

    # A speculation that speech overtook is not worthless: decode only the part
    # it missed and join (see _start_spec_join). Off by default.
    join = _start_spec_join(audio, deadline)
    if join is not None:
        jt, jbox = join
        jt.join(timeout=max(0.0, deadline - _time.monotonic()))
        jtext = jbox.get("text", "")
        if jtext.strip():
            _LAST_FINAL_PATH = "spec-join"
            return (jtext, len(jtext))

    worker, box = _start_final_decode(audio, deadline)
    worker.join(timeout=max(0.0, deadline - _time.monotonic()))
    text = box.get("text", "")
    if text.strip():
        _LAST_FINAL_PATH = "tail-decode"
        return (text, len(text))

    # Only hand back the fallback if it is a TRANSCRIPT rather than a fragment.
    # Non-blank is not the test: the measured disaster returned 0.17 meaning and
    # a fact flip, which is worth less than a final arriving three seconds late.
    fallback = _best_effort_text()
    if fallback.strip() and _best_effort_is_trustworthy(audio):
        _LAST_FINAL_PATH = "best-effort"
        return (fallback, len(fallback))

    # Nothing in hand at all. Returning now would be a BLANK final, which the
    # scorecard scores 0 for the clip — strictly worse than a late one, which
    # still scores its quality (capped 70 past 4s, 50 past 6s). So keep waiting.
    # Stop short of hard_stop so the last resort below still gets a turn. This
    # is NOT the reserve that was measured harmful earlier: that one abandoned a
    # healthy speculation which then went on to land. By this line the decode
    # has already missed the budget AND the long wait, so it is hung, and
    # holding a couple of seconds back costs nothing that was going to arrive.
    last_resort_s = _env_f("DHWANI_LAST_RESORT_S", 2.5)
    worker.join(timeout=max(0.0, min(_env_f("DHWANI_HARD_WAIT_S", HARD_WAIT_S),
                                     hard_stop - last_resort_s - _time.monotonic())))
    if box.get("text", "").strip():
        _LAST_FINAL_PATH = "overrun-wait"
        text = box["text"]
        return (text, len(text))
    text = _best_effort_text()
    if text.strip():
        _LAST_FINAL_PATH = "overrun-wait"
        return (text, len(text))

    # Everything above has now failed, including the fallback that exists so
    # this cannot happen. Two rounds were submitted where exactly this held on
    # every clip, and the engine returned "" each time. A blank is the hardest
    # cap on the card — 0 for the clip, no partial credit — so the correct move
    # is one more decode, of any quality, at any cost, with every optional
    # argument stripped off. Anything it returns is worth more than nothing.
    text = _last_resort_decode(audio, max(0.0, hard_stop - _time.monotonic()))
    _LAST_FINAL_PATH = "last-resort" if text.strip() else "blank"
    return (text, len(text))


def _last_resort_decode(audio: bytes, seconds: float) -> str:
    """The bare minimum: greedy, no prompt, no language, no options at all.

    Deliberately bypasses _decode_final_segment and the whole router — those
    are where the unsupported-argument failures live, and this runs precisely
    when they have all failed. Straight at the backend, cheapest possible call,
    both backends tried in turn.

    Bounded, because the harness drops a clip that never answers (a hard zero,
    strictly worse than a blank). If a decode is hung, this one will hang too;
    `seconds` makes that survivable instead of fatal.
    """
    box: dict[str, str] = {}

    def _run() -> None:
        for backend in ("current", "ctranslate2"):
            try:
                if backend == "ctranslate2" and _resolve_backend() == "ctranslate2":
                    break                  # already tried as "current"
                if backend == "ctranslate2":
                    globals()["_backend"] = "ctranslate2"
                words, lang = _transcribe(audio, None, "", final=False, fast=True)
                text = _normalize_numbers(_deloop(_text_from(words, lang or "en")))
                if text.strip():
                    box["text"] = text
                    return
            except Exception as exc:
                _log_error(f"last-resort decode via {backend} backend", exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=max(0.0, seconds))
    return box.get("text", "")


MIX_MIN_S = 1.0     # don't start the mix decode with less than this left

def _time_for_mix(deadline: float | None) -> bool:
    """Whether there is room for the mix model's extra decode. No deadline
    (speculation, committed windows) means unlimited: those run off the
    critical path and their quality is the whole point."""
    if deadline is None:
        return True
    import time as _time
    return (deadline - _time.monotonic()) >= _env_f("DHWANI_MIX_MIN_S", MIX_MIN_S)


def _final_budget_s() -> float:
    """Wall-clock the final call may spend before giving up on a real decode.

    This was 3.0 for three rounds, on the reasoning that the latency curve pays
    nothing past 5000ms so the final must never approach it. That reasoning read
    the curve and ignored the caps, and it was backwards.

    The caps in streaming_scorecard apply to the TOTAL, not to quality, and they
    are gentle: a final past 4000ms caps the clip at 70, past 6000ms at 50.
    Meanwhile the thing the budget does when it expires — return
    _best_effort_text() — is not a slightly worse transcript, it is the rolling
    LocalAgreement partials, which is the exact text the whole engine was built
    to keep out of the final.

    Measured on the same clip, same machine, two runs minutes apart, the only
    difference being which side of the 3.0s budget it landed on:

        2512ms, real decode      meaning 0.86, no flip   -> clip scores 83.2
        3005ms, best-effort       meaning 0.17, FACT FLIP -> clip scores 24.7

    And the counterfactual, had it simply been allowed to finish:

        real decode at 3500ms -> 75.2      at 5500ms -> 63.2
        real decode at 4500ms -> 69.2      at 6500ms -> 50.0  (worst case)

    Every one of those beats 24.7. The budget was spending ~50 points of quality
    to buy at most 16 points of latency. So it now sits just under the 6000ms
    cap edge, where a late real decode still scores its quality out of 70, and
    it exists only to stop a genuinely hung decode from hanging forever.

    Being fast is still worth 30 points and is still the goal — but it has to
    come from finishing sooner (speculation, a cheaper model), never from
    returning worse text on time.
    """
    try:
        return max(0.2, float(os.environ.get("DHWANI_FINAL_BUDGET_S", "5.4")))
    except ValueError:
        return 5.4


def _start_final_decode(audio: bytes, deadline: float | None = None) -> tuple[threading.Thread, dict]:
    """Start the fast final decode in a worker so the caller can wait on it
    under a deadline. The worker is never killed, only abandoned — if the
    caller gives up, the result simply lands in `box` unread."""
    box: dict[str, str] = {}

    def _run() -> None:
        try:
            box["text"] = _final_decode(audio, fast=True, deadline=deadline)
        except Exception as exc:
            _log_error("final tail decode (falling back to best-effort text)", exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t, box


def _best_effort_is_trustworthy(audio: bytes) -> bool:
    """Is the text we already hold a transcript of the CLIP, or a fragment?

    _best_effort_text() is never blank once anything has decoded, so "did it
    return something" is not the question. The question is whether that
    something covers the audio. Two sources are real decodes with full context —
    the committed windows and a completed speculation — and both report how many
    bytes they account for. The third, the rolling LocalAgreement partials, is a
    1-3s sliding window on a small model and is the text that scored 0.17
    meaning with a fact flip on a clip whose real decode scored 0.86.

    So: trust the fallback when a real decode covers most of the clip, and
    otherwise keep waiting. The scorecard makes that trade obvious — a real
    final at 6500ms still scores 50 for the clip, and the fragment scored 24.7.
    """
    if not audio:
        return True
    with _state_lock:
        covered = max(_fc_bytes, _spec_covered)
    return covered >= _env_f("DHWANI_BE_MIN_COVER", 0.6) * len(audio)


def _best_effort_text() -> str:
    """The best transcript already in hand, no decoding. Never blank if any
    decode has produced anything for this clip."""
    with _state_lock:
        committed_windows = _fc_text.strip()
        live = (_committed + _tail).strip()
        spec, spec_bytes, fc_bytes = (_spec_text or "").strip(), _spec_covered, _fc_bytes
    # A speculation that covered only a PREFIX is unusable as the scored final
    # (_spec_take rejects it), but here the alternative is a blank final, and a
    # blank scores 0 — the hardest cap on the card. It is a whole-buffer decode
    # with full context, so where it reaches further into the clip than the
    # committed windows do, it is also the best text we hold.
    best = spec if (spec and spec_bytes > fc_bytes) else committed_windows
    if best and live:
        return _normalize_numbers(_deloop(_pick_better(best, live)))
    return _normalize_numbers(_deloop(best or live))


def _chunk_s() -> float:
    """Seconds of audio a window may hold before the committer closes it.

    Stays at 12, and the reason is measured. Dropping it to 6 makes windows
    close on ordinary 10s clips and is a large LATENCY win — on the real
    streaming path, 20.51 -> 28.20 of 30 with the median end-to-final at 1024ms
    instead of 1560ms. It is also a real QUALITY loss, which only shows up once
    the committer actually locks something: -6.90/70 mean over six clips, worst
    case a Hinglish clip at 62.1 -> 35.0 with a NEW fact flip, because joined
    segments lose cross-boundary context and _fc_lang pins the language from the
    first closed window, which is wrong on code-switched speech.

    Net that is a wash on points and strictly worse on variance, since a fact
    flip is a hard 50-cap. So 12 stays: on dictation-length clips it commits
    nothing and the final is a clean whole-buffer decode, while long clips still
    get their tail bounded. The latency has to come from somewhere that does not
    cut the audio into pieces.
    """
    try:
        return float(os.environ.get("DHWANI_CHUNK_S", "12") or 0)
    except ValueError:
        return 12.0


def _fc_lang_pin() -> bool:
    """Whether the first closed window PINS the language for the whole clip.

    Pinning is cheap (later windows skip detection) and correct on monolingual
    audio. It is also the named suspect in the DHWANI_CHUNK_S=6 regression: the
    worst clip there was Hinglish, where the first window pinned a language the
    rest of the clip did not keep, and it came back with a NEW fact flip. That
    hypothesis has never been tested separately from the window size that
    exposed it, because on this hardware windows almost never closed at all.
    DHWANI_FC_LANG_PIN=0 re-detects per window and lets the two be measured
    apart. Default ON — unchanged behaviour.
    """
    return _env_flag("DHWANI_FC_LANG_PIN", True)


def _join_final(prefix: str, tail: str) -> str:
    prefix, tail = (prefix or "").strip(), (tail or "").strip()
    if not prefix:
        return tail
    if not tail:
        return prefix
    return f"{prefix} {tail}"


def _final_decode(audio: bytes, prompt: str = "", fast: bool = False,
                  deadline: float | None = None) -> str:
    """Produce the scored final transcript for `audio`.

    Short clip (or DHWANI_CHUNK_S=0): decode the WHOLE buffer fresh. We do NOT
    reuse the LocalAgreement partials, because on the Mac that path hallucinated
    — a Hindi clip came back as Malay ("Terima kasih...") and another as a
    repetition loop, while the same audio decoded whole gave correct Hindi.
    Whisper is trained on 30s context; the 1-3s rolling windows the partials use
    give it too little to anchor on. A whole-buffer decode with fresh language
    detection reproduces the reliable batch result.

    Long clip: the committer has already locked the finished windows into
    `_fc_text` (each decoded as a large ~CHUNK_S window with full context, NOT
    the tiny windows that caused the hallucination above). Here we only decode
    the remaining tail and join. That bounds the end-of-clip decode to one
    window no matter how long the clip is — which is the whole point.
    """
    with _state_lock:
        fc_text, fc_bytes, fc_lang = _fc_text, _fc_bytes, _fc_lang
    if fc_bytes and fc_bytes < len(audio):
        seg = audio[fc_bytes:]
        seg_text, _ = _decode_final_segment(seg, pinned_lang=fc_lang,
                                             prompt=fc_text[-PROMPT_CHARS:], fast=fast,
                                             deadline=deadline)
        text = _join_final(fc_text, seg_text)
        if not text.strip():   # the tail decode produced nothing — keep the prefix
            text = fc_text.strip()
        return _normalize_numbers(_deloop(text))

    seg_text, _ = _decode_final_segment(audio, pinned_lang=None, prompt=prompt, fast=fast,
                                        deadline=deadline)
    if not seg_text.strip():  # last resort: whatever the partials managed to commit
        with _state_lock:
            seg_text = _deloop((_committed + _tail).strip())
    return _normalize_numbers(seg_text)


# Distinct Latin words in the mix model's output that justify SKIPPING the
# primary decode entirely. Deliberately stricter than _mix_latin_min(), which
# only has to arbitrate between two transcripts we already hold: skipping is the
# risky action, so it demands more evidence. Measured on the local corpus, this
# separates cleanly with margin — the three Hinglish clips produced 6, 8 and 6
# Latin tokens, the five pure-Hindi clips 0, 3, 3, 1 and 1.
MIX_FIRST_LATIN_MIN = 5

# Devanagari is the OTHER half of the code-switch test, and it is a safety guard
# rather than a quality one. The hint that triggers mix-first comes from the
# cheap draft model, which can mis-detect an English clip as Hindi; today that
# is harmless (the primary still auto-detects), but skipping the primary on a
# bad hint would not be — zero-stt on English audio returns plenty of Latin and
# would sail past the token threshold. Genuine code-switched speech is Latin AND
# Devanagari; English is Latin only. All eight local Indic clips carry
# Devanagari, so this costs nothing on the audio it is meant to serve.
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def _indic_hint(pinned_lang: str | None) -> bool:
    """Do we already have evidence this clip is Indic, without paying for it?

    Three sources, cheapest-first, and all of them are work already done:
    a committed window pinned the language using the FINAL model (most
    reliable), the streaming partials detected one on the draft model, or —
    when neither field got populated — the text produced so far is visibly
    Devanagari, which is evidence in itself.

    The third source exists because the first two can both be empty: the partial
    worker yields to the committer and the speculator, so on a machine where
    decodes run long relative to the clip it may never complete, and `_lang`
    stays None. Measured here: on all three local Hinglish clips `_lang` was
    still None at the end of the clip. No hint simply means the ordinary
    detect-then-escalate path runs, so this is never worse — but each extra
    source is another clip that gets the cheaper route.

    NB deliberately does NOT call _detect_language(): on the mlx backend that
    runs a full mlx_whisper.transcribe(), i.e. an entire decode, which would
    cost more than the decode it is trying to save.
    """
    if pinned_lang:
        return pinned_lang in _INDIC
    if _lang:
        return _lang in _INDIC
    with _state_lock:
        so_far = _fc_text + _committed + _tail
    return bool(_DEVANAGARI.search(so_far))


def _mix_decode(audio: bytes, prompt: str = "", fast: bool = False) -> str:
    """One decode on DHWANI_MIX_MODEL, through whichever backend it needs."""
    backend = _mix_backend()
    if backend == "mlx":
        converted = _mix_mlx_dir(convert=False)
        if converted:
            try:
                words, _ = _transcribe(audio, "hi", prompt=prompt, final=True,
                                       fast=fast, model=converted)
                return _deloop(_text_from(words, "hi"))
            except Exception as exc:
                # Never lose the clip to the optimisation: same call, slower path.
                _log_error("mix decode via converted mlx — retrying transformers", exc)
        backend = "transformers"       # conversion missing or failed; still correct
    if backend == "transformers":
        words, _ = _transcribe_mix_transformers(audio)
    else:
        words, _ = _transcribe(audio, "hi", prompt=prompt, final=True, fast=fast,
                               model=_mix_model())
    return _deloop(_text_from(words, "hi"))


def _start_primary_decode(audio: bytes, lang: str | None, prompt: str, fast: bool,
                          model: str | None) -> tuple[threading.Thread, dict]:
    """Run the primary decode in a worker so it can overlap the mix decode.
    Never killed, only abandoned — a result nobody reads costs nothing."""
    box: dict = {}

    def _run() -> None:
        try:
            words, raw = _transcribe(audio, lang, prompt=prompt, final=True,
                                     fast=fast, model=model)
            box["words"], box["raw"] = words, raw
        except Exception as exc:
            _log_error("parallel primary decode", exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t, box


def _decode_final_segment(audio: bytes, pinned_lang: str | None,
                          prompt: str = "", fast: bool = False,
                          deadline: float | None = None) -> tuple[str, str | None]:
    """Decode one segment and return (deloop'd text, detected language).

    `pinned_lang` skips auto-detection when the clip's language was already
    decided on an earlier window.

    `fast` drops the temperature-fallback ladder (up to six sequential
    re-decodes for little measured gain). It does NOT drop the mix pass: that
    one costs a single extra decode and is worth ~+28 points/70 on Hinglish,
    because the mix model writes English terms in LATIN and the scorecard greps
    `must_have` terms as Latin substrings. Dropping it once cost 4 of 8 clips
    their facts axis (turbo wrote इंप्रेस/टिटोरिल where gold needs
    impress/tutorial). It is instead skipped only when `deadline` says there is
    no time left for it — quality when affordable, speed when not.
    """
    forced = os.environ.get("DHWANI_LANG")
    if forced:
        # A user override is respected verbatim: single decode, no second-guess.
        try:
            words, _ = _transcribe(audio, forced, prompt=prompt, final=True, fast=fast)
            return _deloop(_text_from(words, forced)), forced
        except Exception as exc:
            _log_error(f"finalize/forced-decode (lang={forced!r})", exc)
            return "", forced

    lang0 = pinned_lang  # None -> let whisper auto-detect on this segment
    text, raw = "", None

    # One pass: auto-detect (or use the pinned language) and transcribe. If this
    # raises (bad repo id, network hiccup, mlx API mismatch), log and fall
    # through to the Hindi retry rather than silently returning blank — a raised
    # exception here used to be indistinguishable from "produced nothing".
    # Language hint from work already done: the committed windows pinned it, or
    # the streaming partials did on the cheap draft model. Costs nothing here.
    hint = pinned_lang or _lang
    primary_model = _en_model() if (hint == "en" and _en_model()) else None
    mix = _mix_model()
    gate = os.environ.get("DHWANI_MIX_GATE", "indic")

    # MIX-FIRST. When the hint already says Indic we know the mix pass is coming,
    # and on code-switched audio its answer wins outright — measured on the three
    # local Hinglish clips, mix-only scored IDENTICALLY to the pair-and-pick
    # (63.2/58.9/54.2 both ways) for 62.6% less decode time. So run it first and,
    # when its output is plainly code-switched, return without ever paying for
    # the primary. That halves the final's cost on exactly the clips whose two
    # sequential decodes make Hindi/Hinglish the slowest finals we produce.
    #
    # Pure Hindi is NOT skipped: there the primary does earn its keep (mix-only
    # measured -5.8/70 mean, worst clip -13.6), so those clips still run both —
    # only in the other order, at the same cost, reusing this candidate below.
    mix_first: str | None = None
    mix_first_cost = 0.0
    primary_thread: threading.Thread | None = None
    primary_box: dict = {}
    if (mix and _indic_hint(pinned_lang) and gate != "codeswitch"
            and _time_for_mix(deadline)):
        import time as _time
        # PARALLEL (DHWANI_MIX_PARALLEL, off by default). Mix-first is a strict
        # win on code-switched clips, where the primary is never run. On PURE
        # Hindi it is only a reordering: both decodes still run, one after the
        # other, and that pair is the slowest final this engine produces.
        #
        # Starting the primary now, alongside the mix decode, makes pure Hindi
        # cost max(mix, primary) instead of mix + primary, and costs nothing on
        # code-switched clips — the mix answer returns and the primary thread is
        # simply abandoned unread, with no other work left to contend with.
        #
        # The open question is whether two decodes on ONE accelerator actually
        # overlap or merely serialise while slowing each other down. That is a
        # property of the scoring host's GPU, not of this code, which is exactly
        # why it ships off and gets measured on Apple hardware.
        if _env_flag("DHWANI_MIX_PARALLEL"):
            primary_thread, primary_box = _start_primary_decode(
                audio, lang0, prompt, fast, primary_model)
        _t0 = _time.monotonic()
        try:
            mix_first = _mix_decode(audio, prompt=prompt, fast=fast)
        except Exception as exc:
            _log_error(f"finalize/mix-first (model={mix!r})", exc)
        mix_first_cost = _time.monotonic() - _t0
        if mix_first and not _looks_bad(mix_first) and _DEVANAGARI.search(mix_first):
            code_switched = len(_latin_tokens(mix_first)) >= MIX_FIRST_LATIN_MIN
            # MIX-ONLY. On code-switched audio the primary was already skipped
            # because the mix answer wins outright. On PURE Hindi it used to run
            # anyway, worth a measured +5.8/70 of quality — and that trade only
            # ever looked good while the clock was invisible.
            #
            # Priced on real hardware it is clearly negative. The primary costs
            # a whole extra decode (1.4s on an M4 Pro, more on the M1 Pro this
            # is scored on), which moves a ~2.1s final to ~3.5s: 23 latency
            # points down to 12. Paying 11 points to buy 5.8 is not a trade
            # worth making, and it gets worse on slower hardware, not better.
            #
            # DHWANI_MIX_ONLY=0 restores the pair, and the sweep measures both.
            if code_switched or _env_flag("DHWANI_MIX_ONLY", True):
                return mix_first, "hi"
        # Not code-switched, so the primary IS worth running on this audio —
        # measured at ~5.8/70 on pure Hindi. But only if it fits.
        #
        # The decode that just ran is a free estimator of what the next one
        # costs: same audio, comparable model. So run the primary only when at
        # least that much budget remains. This is what makes the trade
        # self-calibrating rather than a guess about hardware nobody here can
        # measure — on a fast host both decodes fit and quality wins; on a slow
        # one the primary is dropped, which costs 5.8 of the 70-point quality
        # axis and buys far more than that back on the 30-point latency axis,
        # where halving a 3.5s final is worth about 13 points.
        #
        # Without this, reordering would make a tight budget strictly WORSE than
        # before: the mix decode is now the one that has already run, and the
        # primary below is unconditional.
        if mix_first and deadline is not None:
            left = deadline - _time.monotonic()
            if left < max(mix_first_cost, _env_f("DHWANI_MIX_MIN_S", MIX_MIN_S)):
                return mix_first, "hi"

    try:
        if primary_thread is not None:
            import time as _time
            primary_thread.join(timeout=(None if deadline is None
                                         else max(0.0, deadline - _time.monotonic())))
            if "words" not in primary_box:
                # Still running (or it raised). Leave `text` empty and let the
                # escalation below fall back to mix_first, exactly as it does
                # when the primary decode fails for any other reason.
                raise TimeoutError("parallel primary decode did not finish in budget")
            words, raw = primary_box["words"], primary_box["raw"]
        else:
            words, raw = _transcribe(audio, lang0, prompt=prompt, final=True, fast=fast,
                                     model=primary_model)
        lang = pinned_lang or ("hi" if raw in _INDIC else raw)
        text = _deloop(_text_from(words, lang))
    except Exception as exc:
        _log_error(f"finalize/primary-decode (model={_model_name(True)!r})", exc)

    # Re-decode forcing Hindi when the auto pass mis-fired or flat-out failed:
    # an Indic language that isn't hi (Urdu -> Arabic script -> scores zero), an
    # outright hallucination (loop, blank, a language not in this en+hi corpus),
    # or the primary decode raised (text is still ""). The same pass doubles as
    # the ROUTER's escalation to DHWANI_MIX_MODEL — but ONLY for code-switched
    # audio (Latin letters in the Hindi-detected primary). Measured 2026-07-22
    # on 15 local clips: Apex scored 55.8/70 on Hinglish (meanings 0.92/0.94 vs
    # turbo's ~0.85 on the Mac) but 5.6/70 on pure-Devanagari FLEURS-Hindi —
    # romanized output only survives where romanized gold_alternatives exist,
    # and pure-Hindi rows in the local manifests don't carry them.
    detected = pinned_lang or ("hi" if raw in _INDIC else raw)
    is_indic = (pinned_lang in _INDIC) if pinned_lang else (raw in _INDIC)
    bad = (
        not text
        or (is_indic and detected != "hi")
        or _looks_bad(text)
        or (raw not in ("en", "hi") and raw is not None and pinned_lang is None)
    )
    # Gate: "indic" (default) escalates every Hindi-detected clip — measured
    # safe because zero-stt ties the default model on pure Hindi. "codeswitch"
    # additionally requires Latin letters in the primary; use it for a mix
    # model that (like Apex) collapses on pure-Devanagari audio. NB the
    # codeswitch signal is weak: whisper often writes Hinglish clips entirely
    # in Devanagari, so true code-switch clips can look Latin-free here.
    want_mix = (mix is not None and is_indic
                and (bad or gate != "codeswitch" or _has_latin(text))
                and _time_for_mix(deadline))
    if bad or want_mix:
        try:
            if mix_first is not None and want_mix:
                candidate = mix_first     # already decoded above; never pay twice
            elif want_mix and _mix_backend() == "transformers":
                candidate = _deloop(_text_from(_transcribe_mix_transformers(audio)[0], "hi"))
            else:
                kw = {"model": mix} if (want_mix and mix) else {}
                words_hi, _ = _transcribe(audio, "hi", prompt=prompt, final=True,
                                          fast=fast, **kw)
                candidate = _deloop(_text_from(words_hi, "hi"))
            # Mix models can verbalize digits ("334" -> "3.3 0.4", measured on
            # Apex): if the healthy primary heard a multi-digit number the
            # candidate lost, the facts axis (20 pts, hard 50-cap) outweighs
            # any meaning gain — keep the primary.
            if want_mix and text and not bad and _drops_numbers(text, candidate):
                pass
            elif text:
                text = _pick_mixed(text, candidate) if want_mix else _pick_better(text, candidate)
            else:
                text = candidate
            if is_indic:
                detected = "hi"
        except Exception as exc:
            _log_error(f"finalize/hindi-retry (model={(mix or _model_name(True))!r})", exc)

    return text, detected


def _text_from(words, lang) -> str:
    return "".join(t for (t, _, _) in map_words_triples(words, lang)).strip()


def _pick_better(a: str, b: str) -> str:
    """Choose between two candidate finals with no gold to compare against:
    a clean, longer transcript beats a hallucinated or empty one."""
    ba, bb = _looks_bad(a), _looks_bad(b)
    if ba != bb:
        return b if ba else a
    return b if len(_ntoks(b)) >= len(_ntoks(a)) else a


def _ntoks(text: str) -> list[str]:
    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


_LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def _latin_tokens(text: str) -> set[str]:
    return {t.lower() for t in _LATIN_TOKEN.findall(text or "")}


def _pick_mixed(primary: str, candidate: str) -> str:
    """Choose between the default model's final and the mix model's, for
    code-switched audio, with no gold to compare against.

    Token COUNT is the wrong signal here and cost real points: on the measured
    Hinglish clips turbo's all-Devanagari final was the longer string, so
    _pick_better kept it (26.3/70) over the mix model's (63.2/70). What the
    scorecard actually rewards is English terms surviving in LATIN — must_have
    is grepped as Latin substrings — so prefer whichever candidate preserved
    more distinct Latin words.

    The threshold is the other half of the rule. On pure-Hindi audio the two
    models measured level (51.4 vs 51.3 mean) but disagree clip to clip, so
    swapping on a weak signal is a coin-flip — it lost 13.6 points on one
    FLEURS-Hindi row. Measured separation on the local set is clean: genuinely
    code-switched clips carry 6-8 distinct English words in the mix output
    (document, formatting, impress, tutorial, slide, insert...), while
    pure-Hindi clips carry 0-3, and those are transliterated proper nouns
    ("terrivision", "shipboard") that a pure-Devanagari gold cannot match
    anyway. So require a real English presence before swapping.
    """
    bad_p, bad_c = _looks_bad(primary), _looks_bad(candidate)
    if bad_p != bad_c:
        return candidate if bad_p else primary
    n_p, n_c = len(_latin_tokens(primary)), len(_latin_tokens(candidate))
    return candidate if (n_c > n_p and n_c >= _mix_latin_min()) else primary


def _mix_latin_min() -> int:
    """Distinct English words the mix output must carry before it can replace
    the primary. 4 sits inside a wide measured gap (pure Hindi 0-3, genuine
    code-switch 6-8), not on a knife edge."""
    try:
        return max(1, int(os.environ.get("DHWANI_MIX_LATIN_MIN", "4")))
    except ValueError:
        return 4


def _decode_and_commit(audio: bytes, final: bool, gen: int | None = None) -> None:
    global _committed, _committed_bytes, _tail, _prev_words, _lang

    def _stale() -> bool:
        return gen is not None and gen != _clip_gen

    if _stale():
        return

    with _state_lock:
        start_bytes = _committed_bytes
        prompt = _committed[-PROMPT_CHARS:]
        prev = list(_prev_words)
        lang = _lang

    window = audio[start_bytes:]
    if len(window) < int(_env_f("DHWANI_MIN_DECODE_S", MIN_DECODE_S) * BYTES_PER_SEC) and not final:
        return
    if not window:
        return
    if (_finalizing or _fc_busy) and not final:
        return  # yield: the scored final / committer owns the accelerator

    # Decide the language once, on enough audio, and never commit before then.
    # Whisper called on 0.8s of Hinglish reports "en" and then *translates* the
    # Hindi away; on other clips it reports "ur" and answers in Urdu script.
    if lang is None:
        if len(audio) < int(LANG_DETECT_S * BYTES_PER_SEC) and not final:
            return
        lang = _detect_language(audio, final=final)
        with _state_lock:
            _lang = lang

    window_start_s = start_bytes / BYTES_PER_SEC
    words, _ = _transcribe(window, lang, prompt, final=final)

    with _state_lock:
        if (_finalizing and not final) or _stale():
            return  # a final decode, or a whole new clip, superseded this worker

    words = [(t, s + window_start_s, e + window_start_s) for (t, s, e) in words]

    if final:
        new_text = "".join(t for (t, _, _) in map_words_triples(words, _lang))
        with _state_lock:
            _tail = new_text
            _prev_words = []
        return

    window_end_s = len(audio) / BYTES_PER_SEC
    n_agreed = _common_prefix(prev, words)

    # LocalAgreement-2: two consecutive decodes said the same thing, so lock it.
    # If agreement has stalled and the uncommitted window is getting long, commit
    # anything old enough anyway — a long tail is what makes the final call slow.
    if window_end_s - window_start_s > FORCE_COMMIT_S:
        n_agreed = max(n_agreed, sum(1 for (_, _, e) in words if e < window_end_s - COMMIT_LAG_S))

    while n_agreed > 0 and words[n_agreed - 1][2] >= window_end_s - COMMIT_LAG_S:
        n_agreed -= 1

    lock_words = map_words_triples(words[:n_agreed], _lang)
    rest_words = map_words_triples(words[n_agreed:], _lang)

    with _state_lock:
        if lock_words:
            _committed += "".join(t for (t, _, _) in lock_words)
            _committed_bytes = _even(int(words[n_agreed - 1][2] * BYTES_PER_SEC))
        _tail = "".join(t for (t, _, _) in rest_words)
        _prev_words = words


def _pcm_to_f32(pcm: bytes):
    import numpy as np

    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def _resolve_backend() -> str:
    global _backend
    if _backend is not None:
        return _backend
    forced = os.environ.get("DHWANI_BACKEND", "auto")
    if forced in ("ctranslate2", "mlx", "speechanalyzer"):
        # explicit choices are honored as-is and fail loudly if unavailable.
        # speechanalyzer is NEVER auto-selected — it needs a compiled Swift
        # binary and macOS 26, so it must be opted into deliberately.
        _backend = forced
    else:
        try:
            import mlx_whisper  # noqa: F401
            _backend = "mlx"
        except Exception:
            _backend = "ctranslate2"
    return _backend


def _model_name(final: bool) -> str:
    # The FINAL is a fresh whole-buffer decode (see _finalize) and is the only
    # thing scored. Default is large-v3-turbo, on the 2026-07-17 Mac run's
    # numbers: it BEAT medium on quality (56.28 vs 55.91/70, and clearly on the
    # Hinglish clips: meaning 0.87/0.85 vs 0.81/0.74) and its 4-layer decoder is
    # several times faster than medium's 24 — medium's finals ran 2137-5271ms,
    # sliding down the latency curve; turbo attacks exactly that.
    #
    # The DRAFT model only powers the streaming partials, which are NOT scored
    # and NO LONGER feed the final. So it defaults to a cheap model: it just has
    # to keep the stream alive without contending with the final decode for the
    # GPU. Raising it buys nothing on score.
    if final:
        return os.environ.get("DHWANI_MODEL", "large-v3-turbo")
    return os.environ.get("DHWANI_DRAFT_MODEL", "small")


# Chosen by measurement (2026-07-22, 15 local clips, quality axis of the real
# scorer): zero-stt beat turbo 58.78 vs 37.36 on Hinglish, TIED it on pure
# Hindi (51.36 vs 51.30) — so escalating every Hindi-detected clip is free —
# and outputs native mixed script, so it scores against the PRIMARY gold with
# no dependence on romanized gold_alternatives. Apex (Apache-2.0) measured
# 55.79 Hinglish but 5.60 pure-Hindi: usable via env override, never default.
DEFAULT_MIX_MODEL = "shunyalabs/zero-stt-hinglish"


def _en_model() -> str | None:
    """Optional stronger checkpoint for ENGLISH finals.

    English clips never pay for the mix decode, so they have budget to spare --
    and their losses are rare-word mishearings (alkaline->acolyte, Sie->say,
    Sintra->Cintra) that each cost the whole 20-point facts axis and cap the
    clip at 50. A bigger model is the only thing that fixes those. Unset means
    English stays on DHWANI_MODEL.
    """
    return os.environ.get("DHWANI_EN_MODEL") or None


def _mix_model() -> str | None:
    """Dedicated Hindi/code-switch model for the final's Hindi path.

    With DHWANI_MIX_BACKEND=transformers (the default when the name contains a
    "/"), any HF whisper fine-tune runs as-is via transformers — torch picks
    Apple's MPS on the scoring box, CPU elsewhere. With =native, the name must
    already be in the active backend's format (CTranslate2 dir/repo, or an mlx
    conversion). DHWANI_MIX_MODEL="" disables the mix path entirely.
    """
    value = os.environ.get("DHWANI_MIX_MODEL")
    if value is None:
        value = DEFAULT_MIX_MODEL
    return value or None


def _mix_backend() -> str:
    """Which runtime decodes the mix model.

    Defaults to MLX whenever the primary is on MLX and a converted copy exists,
    because on Apple silicon the transformers/MPS path is the single most
    expensive thing this engine does. Measured on an M4 Pro, per clip, both
    padded to whisper's 30s window so neither scales with clip length:

        zero-stt-hinglish via transformers/MPS   2.16s
        whisper-medium    via mlx                1.08s   <- same size class
        large-v3-turbo    via mlx                1.41s

    zero-stt IS a whisper-medium fine-tune, so that 2x is the runtime, not the
    model. On every Indic clip the mix decode is the whole end-to-final, which
    makes this worth more than any other single change available: 2.16s is 21
    latency points, 1.08s is 30.
    """
    forced = os.environ.get("DHWANI_MIX_BACKEND")
    if forced:
        return forced
    mix = _mix_model()
    if not mix:
        return "native"
    if "/" not in mix:
        return "native"
    if _resolve_backend() == "mlx" and _mix_mlx_dir(convert=False):
        return "mlx"
    return "transformers"


def _mix_mlx_dir(convert: bool = False) -> str | None:
    """Path to an MLX copy of the mix model, converting it if asked.

    Conversion runs ONCE, in warm_models(), while the network is still up — it
    downloads the HF checkpoint and rewrites the weights. It must never happen
    on the scored path: at decode time this only reports a directory that
    already exists, so a missing conversion silently costs speed and never
    correctness (the transformers path still works).

    DHWANI_MIX_MLX_PATH points at a pre-converted directory and skips all of it.
    DHWANI_MIX_MLX=0 disables the whole thing.
    """
    # Default OFF, measured. The conversion itself is byte-correct — what it
    # cannot carry is the DECODING recipe: this fine-tune's generation_config
    # holds 88 suppress_tokens plus begin_suppress and forced_decoder_ids, all
    # applied silently by transformers and all ignored by mlx_whisper. Decoded
    # without them the model repeats itself ("... तो तो ..."), which is slower
    # (more tokens) and worse at once: 2230ms/57.7q -> 4835ms/55.7q on the same
    # clip, same machine. A weight converter cannot fix a sampler mismatch.
    if not _env_flag("DHWANI_MIX_MLX", False):
        return None
    explicit = os.environ.get("DHWANI_MIX_MLX_PATH")
    if explicit:
        return explicit if os.path.isdir(explicit) else None
    repo = _mix_model()
    if not repo or "/" not in repo:
        return None

    cache = os.environ.get("DHWANI_MLX_CACHE") or os.path.join(
        os.path.expanduser("~"), ".cache", "dhwani-mlx")
    out = os.path.join(cache, repo.replace("/", "--"))
    # The VERIFIED marker is written by warm_models only after a REAL decode
    # through the converted weights succeeds. Weights alone are not enough: a
    # conversion that writes plausible tensors mlx cannot actually run would
    # otherwise fail on the first scored clip instead of at warm time.
    has_weights = (os.path.exists(os.path.join(out, "weights.safetensors"))
                   or os.path.exists(os.path.join(out, "weights.npz")))
    if os.path.isdir(out) and has_weights and \
            os.path.exists(os.path.join(out, "VERIFIED")):
        return out
    if not convert:
        return None
    return _convert_mix_to_mlx(repo, out)


def _convert_mix_to_mlx(repo: str, out: str) -> str | None:
    """Run mlx_whisper's converter. Every failure mode ends in None, which means
    'use transformers' — this is an optimisation, never a dependency.

    The CLI's flag names have moved between mlx_whisper releases, so both known
    spellings are tried before giving up. Deliberately a subprocess: conversion
    loads the torch checkpoint and the MLX arrays at the same time, and doing
    that inside the server process would leave both resident for the whole run.
    """
    import subprocess
    import sys

    # solution/hf_to_mlx.py, ours. The obvious tool — `python -m
    # mlx_whisper.convert` — does not exist: the PyPI wheel ships no converter
    # module at all, which is why the first Apple run silently stayed on
    # transformers. Learned by listing the wheel's contents, not from any error
    # that surfaced on its own.
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cmd = [sys.executable, "-m", "solution.hf_to_mlx", repo, out]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800,
                              cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    except Exception as exc:      # noqa: BLE001
        _log_note(f"mix->mlx conversion could not start ({type(exc).__name__}: {exc})")
        return None
    if os.path.exists(os.path.join(out, "weights.safetensors")):
        _log_note(f"mix model converted to MLX at {out}")
        return out
    _log_note("mix->mlx conversion failed: "
              f"{(proc.stderr or proc.stdout or '')[-300:].strip()}")
    return None


# --- transformers mix backend — runs any HF whisper fine-tune unconverted. ---

_mix_state: tuple | None = None
_mix_load_lock = threading.Lock()


def _get_mix_transformers(repo: str):
    global _mix_state
    with _mix_load_lock:
        if _mix_state is None or _mix_state[0] != repo:
            import torch
            from transformers import AutoProcessor, WhisperForConditionalGeneration
            use_mps = bool(getattr(torch.backends, "mps", None)
                           and torch.backends.mps.is_available())
            device = "mps" if use_mps else "cpu"
            dtype = torch.float16 if use_mps else torch.float32
            proc = AutoProcessor.from_pretrained(repo)
            model = WhisperForConditionalGeneration.from_pretrained(
                repo, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
            _mix_state = (repo, proc, model, device)
    return _mix_state


def _mix_beam() -> int:
    """Beam width for the mix model. GREEDY by default, and unlike the primary
    that is the measured answer, not an oversight.

    Beam is nearly free on the primary (encoder-dominated: 22.25s vs 24.12s) and
    bought +4.97/70 on English. On this model, through transformers, it is not
    free and it does not pay:

        Hinglish   58.78/70 @ 52.2s greedy  ->  59.34/70 @ 139.5s at beam 5
        pure Hindi 51.36/70            ->  53.37/70

    The Hinglish clips are the ones whose final IS this model's output, and
    there beam buys +0.56 for 2.7x the decode — on exactly the clips whose
    latency mix-first just halved. The +2.02 on pure Hindi is unreachable:
    _pick_mixed can never return the mix candidate there (it needs
    _mix_latin_min() Latin words and pure-Hindi output carries 0-3), so that
    gain would be computed and thrown away.
    """
    try:
        return max(1, int(os.environ.get("DHWANI_MIX_BEAM", "1")))
    except ValueError:
        return 1


def _transcribe_mix_transformers(window: bytes):
    """Whole-buffer decode on the mix model. Returns the same (words, lang)
    shape as the other backends: one pseudo-word spanning the clip — the final
    path only needs text. Chunks >30s audio (whisper's encoder window) with the
    stitched texts concatenated."""
    import numpy as np
    import torch

    repo = _mix_model()
    if not repo:
        return [], None
    _, proc, model, device = _get_mix_transformers(repo)
    pcm = np.frombuffer(window, dtype=np.int16).astype(np.float32) / 32768.0
    if pcm.size == 0:
        return [], None

    chunk = 30 * SR
    texts = []
    for start in range(0, len(pcm), chunk):
        piece = pcm[start:start + chunk]
        if len(piece) < int(0.2 * SR):
            continue
        inputs = proc(piece, sampling_rate=SR, return_tensors="pt")
        feats = inputs["input_features"].to(device=device, dtype=model.dtype)
        # Beam width matters MORE here than on the primary. This model's output
        # is not a candidate any more — since mix-first it IS the final on every
        # code-switched clip — and it had been decoding greedily while the
        # primary got beam 5 (worth +4.97/70 on the English set). Whisper decode
        # cost is dominated by the fixed encoder pass, so a wider beam is close
        # to free: measured 22.25s vs 24.12s greedy on the primary.
        beams = _mix_beam()
        attempts: list[dict] = [{"task": "transcribe"}, {}]
        if beams > 1:
            attempts.insert(0, {"task": "transcribe", "num_beams": beams})
        # Same lesson as the MLX ladder, applied before it costs anything: a
        # generate() kwarg this transformers version dislikes must degrade the
        # call, not delete the transcript. `task` in particular has moved
        # between generate() and the processor across transformers majors.
        ids, last = None, None
        for kwargs in attempts:
            try:
                with torch.inference_mode():
                    ids = model.generate(feats, **kwargs)
                break
            except Exception as exc:      # noqa: BLE001 — next attempt handles it
                last = exc
                _log_error(f"mix generate rejected kwargs={sorted(kwargs)}", exc)
        if ids is None:
            raise last if last is not None else RuntimeError("mix generate failed")
        texts.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
    text = " ".join(t for t in texts if t).strip()
    dur = len(window) / BYTES_PER_SEC
    return ([(text, 0.0, dur)] if text else []), "hi"


# mlx-community's repo naming is NOT a clean pattern. Some sizes are published
# bare ("whisper-medium", "whisper-large-v3-turbo"); others only exist with an
# "-mlx" suffix ("whisper-small-mlx", "whisper-large-v3-mlx"). The old code
# guessed f"whisper-{size}" for everything, which 401'd on small and large-v3 —
# that was the silent 0.00 on the Mac. Every id below was checked against the
# Hub; the two marked PROVEN also produced real scores on the target Mac, so
# they are deliberately left as-is rather than "tidied" to the -mlx form.
_MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium",             # PROVEN: 41.62/70 on Mac
    # large-v3 history, all learned on the target Mac:
    #   whisper-large-v3-mlx      (3.08 GB fp16 npz)  -> download stalls, rate-limited
    #   whisper-large-v3-8bit     (1.65 GB safetensors) -> downloads fine, but the
    #       installed mlx_whisper only loads weights.npz: "[load_npz] Input must
    #       be a zip file". Format mismatch, not corruption.
    #   whisper-large-v3-mlx-8bit (1.59 GB weights.npz) -> small AND npz. Current.
    "large": "mlx-community/whisper-large-v3-mlx-8bit",
    "large-v3": "mlx-community/whisper-large-v3-mlx-8bit",
    "large-v3-fp16": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",  # PROVEN on Mac
    "turbo": "mlx-community/whisper-large-v3-turbo",
}


def _mlx_repo(name: str) -> str:
    """Map a whisper size to a real mlx-community repo id. A full repo id
    (contains "/") passes through untouched, so DHWANI_MODEL can name any repo."""
    if "/" in name:
        return name
    if name in _MLX_REPOS:
        return _MLX_REPOS[name]
    return f"mlx-community/whisper-{name}-mlx"  # -mlx is the dominant pattern


def warm_models() -> None:
    """Force both models to load now, before the harness blocks the network.

    Does NOT raise: warming is an optimization (pay the load cost once, up
    front, instead of on the first real clip), not a correctness requirement.
    If it fails, log it loudly and move on — every real draft()/_finalize()
    call already survives a decode failure on its own (see _finalize and the
    top-level try/except in draft()), so one bad model shouldn't stop a whole
    run from at least attempting every clip and reporting what happened.
    """
    backend = _resolve_backend()
    if backend == "speechanalyzer":
        try:
            _sa_server()  # spawn the persistent helper and block on READY
        except Exception as exc:
            _log_error("warm_models/speechanalyzer — continuing anyway", exc)
        return
    names = [_model_name(True), _model_name(False)]
    if _mix_model():
        # Convert the mix model to MLX now, while the network is up and nothing
        # is being timed. At scoring time HF_HUB_OFFLINE is set and the final is
        # on the clock, so this is the only moment it can happen.
        converted = None
        if backend == "mlx":
            try:
                converted = _mix_mlx_dir(convert=True)
                if converted and not os.path.exists(os.path.join(converted, "VERIFIED")):
                    converted = _verify_mix_mlx(converted)
            except Exception as exc:
                _log_error("warm_models/mix->mlx conversion — using transformers", exc)
                converted = None
        if converted:
            names.append(converted)
        elif _mix_backend() == "transformers":
            try:
                _get_mix_transformers(_mix_model())   # downloads + loads once
            except Exception as exc:
                _log_error(f"warm_models/mix:{_mix_model()} — continuing anyway", exc)
        else:
            names.append(_mix_model())
    for name in dict.fromkeys(names):
        try:
            if backend == "mlx":
                _transcribe_mlx(b"\x00\x00" * int(0.5 * SR), lang="en", prompt="",
                                 model_name=name, final=False)
            else:
                _get_model_ctranslate2(name)
        except Exception as exc:
            _log_error(f"warm_models/{name} — continuing anyway", exc)
    _verify_final_path()


def _verify_mix_mlx(converted: str) -> str | None:
    """Bless the converted mix model only if it TRANSCRIBES WELL, on real audio.

    The first version of this verify decoded half a second of silence and
    checked for an exception. It passed — and the model it blessed was
    degraded, not broken: correct weights decoded without the fine-tune's
    suppress-token recipe ramble ("... तो तो ..."), which cost time and quality
    at once on the scored path. "Runs without raising" is a claim about the
    graph; the clip is scored on the TEXT.

    So the bar is now a real Hindi clip from the shipped dev set, and three
    checks that all target the observed failure: the output carries Devanagari,
    _deloop finds nothing to remove (a rambling decode is exactly what it
    removes), and the decode lands under a wall-clock bound. No reference clip
    on this machine means NO marker — an optimisation that cannot prove itself
    stays off, because the fallback it displaces is known-good.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wav = os.path.join(root, "data", "dev", "audio", "fleurs_hi_in_test_1666.wav")
    if not os.path.exists(wav):
        _log_note("mix mlx verify: no reference clip on this machine — staying on "
                  "transformers (the conversion is unproven here, not broken)")
        return None
    import time as _time
    import wave as _wave
    with _wave.open(wav, "rb") as w:
        pcm = w.readframes(w.getnframes())
    t0 = _time.monotonic()
    words, _ = _transcribe_mlx(pcm, lang="hi", prompt="", model_name=converted,
                               final=False)
    took = _time.monotonic() - t0
    text = _text_from(words, "hi").strip()
    bound = _env_f("DHWANI_MIX_MLX_VERIFY_S", 4.0)
    problems = []
    if not _DEVANAGARI.search(text):
        problems.append("no Devanagari in the transcript")
    if _deloop(text) != text:
        problems.append("repetition loop (the measured ramble)")
    if took > bound:
        problems.append(f"decode took {took:.1f}s > {bound:.1f}s bound")
    if problems:
        _log_note(f"mix mlx verify REJECTED the conversion: {'; '.join(problems)} "
                  "— staying on transformers")
        return None
    with open(os.path.join(converted, "VERIFIED"), "w") as fh:
        fh.write(f"real-clip decode ok in {took:.1f}s\n")
    _log_note(f"mix mlx conversion verified on a real clip in {took:.1f}s at {converted}")
    return converted


def _verify_final_path() -> None:
    """Prove the SCORED decode path runs on this machine, before a clip does.

    Two submitted rounds scored zero because it did not. The mlx backend is
    selected automatically whenever mlx_whisper imports, was never executable on
    any machine available here, and the first evidence that it did not work
    arrived as an email from the organisers weeks later. A warm-up that only
    loads models cannot catch that: the models loaded fine. What failed was the
    call SHAPE of the final decode.

    So run the real thing once — the same _transcribe(final=True) the scored
    path uses — on a scrap of audio. A blank result is fine and expected on
    noise; the question being asked is only whether it RAISES. If it does, the
    backend cannot produce finals here, and a slow transcript beats no
    transcript by the entire value of the clip, so drop to CTranslate2 for the
    rest of the process.

    DHWANI_VERIFY=0 disables it. Never raises: a broken self-check must not be
    worse than no self-check.
    """
    global _backend
    if not _env_flag("DHWANI_VERIFY", True) or _resolve_backend() != "mlx":
        return
    import numpy as np
    probe = (np.sin(np.arange(int(1.0 * SR)) * 0.07) * 6000).astype(np.int16).tobytes()
    try:
        _transcribe(probe, None, "", final=True, fast=False)
        return
    except Exception as exc:
        _log_error("SELF-CHECK: the mlx final path raised — falling back to "
                   "ctranslate2 for this process. Finals will be slower and "
                   "will not be blank", exc)
    try:
        _backend = "ctranslate2"
        _get_model_ctranslate2(_model_name(True))
    except Exception as exc:
        _log_error("SELF-CHECK: ctranslate2 fallback also failed — leaving the "
                   "backend as-is and relying on the per-decode ladders", exc)
        _backend = "mlx"


def _detect_language(audio: bytes, final: bool) -> str | None:
    forced = os.environ.get("DHWANI_LANG")
    if forced:
        return forced
    backend = _resolve_backend()
    if backend == "speechanalyzer":
        return _sa_locale().split("-")[0]   # SpeechAnalyzer picks its own model
    if backend == "mlx":
        return _detect_language_mlx(audio)
    return _detect_language_ctranslate2(audio)


# Whisper's own anti-hallucination mechanism: if a decode comes out too
# repetitive (high compression ratio) or low-confidence, it retries at a higher
# temperature. Setting a single temperature=0.0 DISABLES this — which is why the
# streaming path fell into "अद्राद अद्राद..." loops and Malay hallucinations.
# The final decode re-enables the fallback ladder; partials stay greedy for speed.
_FINAL_TEMPS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def _final_temps() -> tuple:
    """The temperature ladder for the quality final, as a sweepable list.

    Worth measuring rather than assuming, because it cuts both ways and both
    sides are large. When whisper's thresholds fire, the ladder re-decodes the
    same audio once per temperature — up to six sequential decodes, and the
    speculation that runs it is joined for the WHOLE remaining budget by
    _finalize, so a ladder that fires lands directly on the 30-point latency
    axis. Shortening it is the single biggest latency lever that does not
    change a model.

    Against that, the ladder is whisper's own anti-hallucination mechanism, and
    removing it is what put the streaming path into "अद्राद अद्राद..." loops.
    A repetition loop is a hard 30-point cap on its clip.

    DHWANI_FINAL_TEMPS="0.0,0.4" sets a shorter one; "0.0" disables it entirely.
    """
    raw = os.environ.get("DHWANI_FINAL_TEMPS")
    if not raw:
        return _FINAL_TEMPS
    try:
        temps = tuple(float(x) for x in raw.split(",") if x.strip() != "")
        return temps or _FINAL_TEMPS
    except ValueError:
        return _FINAL_TEMPS


def _transcribe(window: bytes, lang: str | None, prompt: str, final: bool = False,
                model: str | None = None, fast: bool = False):
    backend = _resolve_backend()
    if backend == "speechanalyzer":
        return _transcribe_speechanalyzer(window, final=final)
    audio = _pcm_to_f32(window)
    if audio.size == 0:
        return [], None
    name = model or _model_name(final)
    # `fast` = final-quality model, but greedy: no temperature ladder. The
    # ladder re-decodes the segment once per temperature when whisper's own
    # thresholds fire, so it can multiply wall-clock several times over.
    quality = final and not fast
    if backend == "mlx":
        return _transcribe_mlx(window, lang, prompt, model_name=name, final=quality)
    return _transcribe_ctranslate2(audio, lang, prompt, model_name=name, final=quality)


# --- ctranslate2 (faster-whisper) backend — CPU-only, verified on this box --

def _detect_language_ctranslate2(audio: bytes) -> str | None:
    """Compare total Indic mass against English rather than trusting the
    argmax. Margins are thin on code-switched audio — one sample clip lands
    hi=0.27 vs en=0.10 with the rest of the mass spread over ur/ne/mr — and
    reading "en" is unrecoverable, because whisper then translates the Hindi
    away."""
    model, lock = _get_model_ctranslate2(_model_name(final=True))
    try:
        with lock:
            lang, _prob, all_probs = model.detect_language(_pcm_to_f32(audio))
    except Exception:
        return None
    probs = dict(all_probs or [])
    if sum(probs.get(code, 0.0) for code in _INDIC) >= probs.get("en", 0.0):
        return "hi"
    return "hi" if lang in _INDIC else lang


def _beam_size(final: bool) -> int:
    """Beam width for the scored final. GREEDY, and that is now a correctness
    requirement rather than a speed/quality trade.

    History, because the number in this docstring used to say 5 and the reason
    it did was real: measured on the English clips, beam 5 recovered a rare word
    greedy misheard ("alkaline" -> "acolyte"), 53.26 -> 58.23 of 70 with one
    fewer fact flip. That measurement stands. It was taken on CTranslate2, on a
    Windows box.

    It was then shipped into a path that only ever runs on MLX, which no machine
    here could execute. builderr's runtime does not support beam search in
    mlx_whisper, and the result was not a slower decode or a worse one — it was
    a BLANK final on every clip, across two submitted rounds, both of which
    scored zero and neither of which we could see. Their words: "the default MLX
    path requests beam search, which the scoring runtime does not support, so
    every final came back blank."

    So: greedy everywhere by default. Beam on MLX is separately opt-in via
    DHWANI_MLX_BEAM and still passes through the capability ladder in
    _transcribe_mlx, so even asking for it explicitly cannot blank a decode.
    The English quality beam was buying has to come from somewhere that works on
    the backend being scored — a stronger checkpoint for English finals
    (DHWANI_EN_MODEL), which is a model choice and not an API gamble.
    """
    if not final:
        return 1
    try:
        return max(1, int(os.environ.get("DHWANI_BEAM", "1")))
    except ValueError:
        return 1


def _transcribe_ctranslate2(audio, lang, prompt, model_name, final=False):
    model, lock = _get_model_ctranslate2(model_name)
    extra = dict(
        temperature=list(_final_temps()),
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
    ) if final else dict(temperature=0.0)
    with lock:
        segments, info = model.transcribe(
            audio,
            language=lang,
            task="transcribe",
            beam_size=_beam_size(final),
            condition_on_previous_text=False,
            word_timestamps=True,
            initial_prompt=prompt or None,
            **extra,
        )
        segs = list(segments)          # a generator: materialise before reuse
        words = [
            (w.word, w.start, w.end)
            for seg in segs
            for w in (seg.words or [])
        ]
        if not words:
            # Same silent-blank hazard as the mlx path: word timings can come
            # back empty while seg.text holds a perfectly good transcript, and
            # an empty word list becomes an empty final, which scores zero.
            text = "".join(seg.text or "" for seg in segs).strip()
            if text:
                words = [(text, 0.0, max(0.1, len(audio) / SR))]
    return words, getattr(info, "language", None)


def _get_model_ctranslate2(name: str):
    """Load (once) and return the CTranslate2 model plus its serialising lock.

    `device="auto"` can select a CUDA runtime that then fails to load its
    libraries (a machine with a GPU but no cuBLAS raises
    "Library cublas64_12.dll is not found"). That used to happen on every call,
    since a failed load never populates the cache — so each decode retried it
    and every final came back blank. Fall back to CPU once and remember it.
    """
    from faster_whisper import WhisperModel

    with _registry_lock:
        if name not in _models:
            # Default "cpu", not "auto": CTranslate2 has no Metal backend, so on
            # the Apple-silicon scoring box "auto" is CPU anyway (there the mlx
            # backend does the real work), while elsewhere "auto" can select a
            # CUDA runtime whose libraries then fail mid-decode — which blanked
            # every final on a machine with a GPU but no cuBLAS.
            requested = os.environ.get("DHWANI_DEVICE", "cpu")
            attempts = [requested] if requested == "cpu" else [requested, "cpu"]
            last: Exception | None = None
            for device in attempts:
                compute = "int8" if device in ("cpu", "auto") else "float16"
                try:
                    _models[name] = WhisperModel(
                        name,
                        device=device,
                        compute_type=compute,
                        cpu_threads=os.cpu_count() or 4,
                    )
                    _locks[name] = threading.Lock()
                    break
                except Exception as exc:   # noqa: BLE001 - try the next device
                    last = exc
                    _log_error(f"ctranslate2 load {name!r} on device={device!r}", exc)
            else:
                raise last if last else RuntimeError(f"cannot load {name}")
        return _models[name], _locks[name]


# --- mlx backend — Apple GPU/ANE. UNVERIFIED, no Apple Silicon available. ---
#
# Written from mlx_whisper's published transcribe() shape (a near-mirror of
# openai-whisper's transcribe(), by design, for drop-in compatibility):
# accepts a str path OR a float32 numpy array directly; word_timestamps=True
# returns segments[i]["words"] as {"word","start","end"} dicts. The internal
# model cache is keyed by path_or_hf_repo, so repeat calls with the same repo
# id reuse loaded weights without reloading. None of this has been executed
# against the real package — run tools/mac_smoke_test.py first.

def _get_lock_mlx(repo: str) -> threading.Lock:
    with _registry_lock:
        if repo not in _locks:
            _locks[repo] = threading.Lock()
        return _locks[repo]


# Which rung of the kwarg ladder below is known to work, per repo. Probed once
# on first use and then reused, so the cost is one bad call per model per
# process rather than one per clip.
_MLX_LEVEL: dict[str, int] = {}
_MLX_LEVEL_LOCK = threading.Lock()


def _mlx_kwarg_levels(repo: str, lang: str | None, prompt: str, final: bool,
                      beam: int) -> list[dict]:
    """Progressively smaller argument sets for mlx_whisper.transcribe().

    Rung 0 is everything we want. Each rung drops the next most likely thing to
    be unsupported, and the last two are the calls the function cannot exist
    without. The ladder always has the same SHAPE regardless of `final` or
    `beam`, so a rung index cached for one call means the same thing for the
    next — that is why rung 0 is allowed to be identical to rung 1 when no beam
    was requested, instead of being deduplicated away.

    This exists because the previous "fallback" re-sent the kwargs that had just
    failed, from inside the `except` block that caught them. One unsupported
    keyword therefore took down every decode in the process: the speculation,
    the committed windows, the tail decode AND the partials — which is why
    _best_effort_text() had nothing left to fall back to and the final came back
    blank rather than merely degraded.
    """
    full: dict = dict(path_or_hf_repo=repo, language=lang, task="transcribe",
                      condition_on_previous_text=False, word_timestamps=True,
                      initial_prompt=prompt or None)
    if final:
        full.update(temperature=_final_temps(), compression_ratio_threshold=2.4,
                    logprob_threshold=-1.0, no_speech_threshold=0.6)
    else:
        full.update(temperature=0.0)

    with_beam = dict(full, beam_size=beam) if beam > 1 else dict(full)
    no_thresholds = {k: v for k, v in full.items()
                     if k not in ("compression_ratio_threshold", "logprob_threshold",
                                  "no_speech_threshold", "condition_on_previous_text")}
    no_prompt = {k: v for k, v in no_thresholds.items() if k != "initial_prompt"}
    # Dropping word_timestamps is survivable: the text is still there, and
    # _mlx_words() below synthesises one span from it. Losing the timings costs
    # the partials their word-level agreement, never the scored final.
    no_words = {k: v for k, v in no_prompt.items() if k != "word_timestamps"}
    return [with_beam, full, no_thresholds, no_prompt, no_words,
            dict(path_or_hf_repo=repo, language=lang, task="transcribe"),
            dict(path_or_hf_repo=repo)]


def _mlx_words(result: dict, duration_s: float) -> list:
    """Word spans from an mlx_whisper result, and NEVER an empty list when the
    model actually produced text.

    The old code read segments[i]["words"] and stopped there. An mlx_whisper
    build that returns segments without a "words" key — or that ignored
    word_timestamps, or had it stripped by the ladder above — yielded [], which
    _text_from turned into "", which is a BLANK final scoring zero. With the
    transcript sitting in result["text"] the whole time. No exception, no log
    line, nothing to notice: the single most expensive way for this engine to
    fail. Timings only drive the unscored partials, so a whole-clip pseudo-span
    is a complete substitute for the one thing that is scored.
    """
    words = [(w["word"], w["start"], w["end"])
             for seg in (result.get("segments") or [])
             for w in (seg.get("words") or [])
             if isinstance(w, dict) and "word" in w]
    if words:
        return words
    text = ""
    for seg in (result.get("segments") or []):
        text += seg.get("text") or ""
    text = (text or result.get("text") or "").strip()
    return [(text, 0.0, max(0.1, duration_s))] if text else []


def _mlx_transcribe_raw(pcm, repo: str, lang: str | None, prompt: str,
                        final: bool, beam: int) -> dict:
    """Call mlx_whisper.transcribe(), walking down the ladder until one works."""
    import mlx_whisper

    levels = _mlx_kwarg_levels(repo, lang, prompt, final, beam)
    start = _MLX_LEVEL.get(repo, 0)
    last: BaseException | None = None
    for i in range(start, len(levels)):
        if i > start and levels[i] == levels[i - 1]:
            continue          # rung 0 == rung 1 whenever no beam was requested
        try:
            result = mlx_whisper.transcribe(pcm, **levels[i])
        except Exception as exc:      # noqa: BLE001 — the next rung IS the handler
            # Deliberately catches everything, not TypeError. The guard this
            # replaces caught TypeError only, and the failure that cost two
            # rounds was mlx_whisper refusing beam search with a different
            # exception class entirely — which sailed straight past it and out
            # of the function. A rejected rung is expected traffic, so it logs
            # one line rather than a traceback; total failure below is loud.
            last = exc
            _log_note(f"transcribe_mlx/{repo}: rung {i} rejected "
                      f"({type(exc).__name__}: {str(exc)[:90]}) — trying a smaller call")
            continue
        if i != start:
            with _MLX_LEVEL_LOCK:
                _MLX_LEVEL[repo] = i          # remember, so this costs once
        return result
    raise last if last is not None else RuntimeError(f"mlx_whisper refused every call for {repo}")


def _detect_language_mlx(audio: bytes) -> str | None:
    """mlx_whisper's public transcribe() only surfaces the argmax language
    (result["language"]), not the per-language probability list faster-whisper
    gives us — so the Indic-vs-English mass tiebreak above isn't available
    here. Falls back to the plain argmax, collapsed onto Hindi for any Indic
    language, same as the pre-tiebreak ctranslate2 behaviour."""
    import numpy as np

    repo = _mlx_repo(_model_name(final=True))
    lock = _get_lock_mlx(repo)
    try:
        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        with lock:
            result = _mlx_transcribe_raw(pcm, repo, lang=None, prompt="",
                                         final=False, beam=1)
        lang = result.get("language")
    except Exception as exc:
        _log_error(f"detect_language_mlx/{repo}", exc)
        return None
    return "hi" if lang in _INDIC else lang


def _transcribe_mlx(window: bytes, lang: str | None, prompt: str, model_name: str, final=False):
    import numpy as np

    repo = _mlx_repo(model_name)
    lock = _get_lock_mlx(repo)
    pcm = np.frombuffer(window, dtype=np.int16).astype(np.float32) / 32768.0
    if pcm.size == 0:
        return [], None

    # Beam on MLX is opt-in and separate from DHWANI_BEAM, which now defaults to
    # greedy. Even when asked for, it is only the top rung of the ladder: an
    # unsupported beam_size costs one rejected call, not the decode.
    beam = _beam_size(True) if (final and _env_flag("DHWANI_MLX_BEAM")) else 1
    with lock:
        result = _mlx_transcribe_raw(pcm, repo, lang, prompt, final, beam)
    return _mlx_words(result, len(window) / BYTES_PER_SEC), result.get("language")


# --- speechanalyzer backend — Apple's on-device SpeechAnalyzer via a Swift CLI.
#
# EXPERIMENTAL, macOS 26 + Apple silicon only, opt-in (DHWANI_BACKEND=speechanalyzer).
# Talks to speechanalyzer/speechanalyzer_cli (build it with
# tools/build_speechanalyzer.sh). Only the final is scored, so this backend does
# NOTHING on streaming partials and transcribes the whole buffer once on the
# final — a natural fit for SpeechAnalyzer's file API, which runs faster than
# realtime. The helper is kept warm as a persistent `--serve` process so the
# per-clip cost is just the transcription, not process startup.
#
# Measure quality with tools/speechanalyzer_eval.py BEFORE trusting this; see
# speechanalyzer/README.md for why Hindi output quality is the open question.

import subprocess  # noqa: E402
import tempfile  # noqa: E402
import wave  # noqa: E402

_sa_proc = None
_sa_lock = threading.Lock()


def _sa_locale() -> str:
    return os.environ.get("DHWANI_SA_LOCALE", "hi")


def _sa_cli_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (here, os.path.dirname(here)):  # solution/ or project root
        cand = os.path.join(base, "speechanalyzer", "speechanalyzer_cli")
        if os.path.exists(cand):
            return cand
    return os.path.join(here, "speechanalyzer", "speechanalyzer_cli")


def _sa_server():
    """Start (once) and return the persistent SpeechAnalyzer CLI process,
    blocking until it prints READY so the first transcribe doesn't race setup."""
    global _sa_proc
    if _sa_proc is not None and _sa_proc.poll() is None:
        return _sa_proc
    cli = _sa_cli_path()
    if not os.path.exists(cli):
        raise FileNotFoundError(
            f"speechanalyzer_cli not built at {cli} — run tools/build_speechanalyzer.sh")
    _sa_proc = subprocess.Popen(
        [cli, "--serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, encoding="utf-8", bufsize=1)
    ready = _sa_proc.stdout.readline().strip()
    if ready != "READY":
        raise RuntimeError(f"speechanalyzer_cli did not signal READY (got {ready!r})")
    return _sa_proc


def _write_temp_wav(pcm: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="dhwani_sa_")
    os.close(fd)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)
    return path


def _transcribe_speechanalyzer(window: bytes, final: bool):
    # Partials are unscored and no longer feed the final — skip them entirely so
    # streaming doesn't spawn a transcription per 500ms tick.
    if not final:
        return [], None
    if not window:
        return [], None

    locale = _sa_locale()
    wav = _write_temp_wav(window)
    try:
        with _sa_lock:
            proc = _sa_server()
            proc.stdin.write(f"{wav}\t{locale}\n")
            proc.stdin.flush()
            line = proc.stdout.readline()
        text = (line or "").rstrip("\n").replace("\\n", "\n").strip()
    finally:
        try:
            os.remove(wav)
        except OSError:
            pass
    if not text:
        return [], locale.split("-")[0]
    dur = len(window) / BYTES_PER_SEC
    return [(text, 0.0, dur)], locale.split("-")[0]


# --- speculative finalization ---------------------------------------------
#
# The scored final is a whole-buffer decode; end-to-final latency is how long
# that decode takes AFTER `end` arrives. But people stop talking before they
# release the dictation key, and the evaluator's clips end after the speech
# does — so the decode can usually run DURING the trailing silence instead.
# The speculation produces byte-for-byte the same text the normal path would
# (it calls the same _final_decode on the same buffer); its only failure mode
# is being unusable because speech resumed, in which case the normal decode
# runs and the clip merely scores today's latency instead of ~0 ms.

def _speculation_enabled() -> bool:
    return os.environ.get("DHWANI_SPECULATE", "1") != "0"


def _spec_alive() -> bool:
    t = _spec_thread
    return t is not None and t.is_alive()


def _frame_rms(audio: bytes):
    """RMS per 20 ms frame over the whole buffer, or None without numpy/audio."""
    try:
        import numpy as np
    except Exception:
        return None
    pcm = np.frombuffer(audio[: (len(audio) // 2) * 2], dtype=np.int16)
    frame = int(FRAME_S * SR)
    if pcm.size < frame:
        return None
    pcm = pcm[: (pcm.size // frame) * frame].astype(np.float32)
    frames = pcm.reshape(-1, frame)
    return np.sqrt((frames * frames).mean(axis=1))


def _silence_threshold(rms) -> float:
    """Gain-invariant: the evaluator replays each clip at deliberately different
    gains, so the threshold hangs off the loudest frame this clip has produced,
    with a small absolute floor for digitally-silent zeros."""
    return max(60.0, 0.05 * float(rms.max()))


def _tail_is_silent(rms, thr) -> bool:
    n = max(1, int(_env_f("DHWANI_SPEC_SILENCE_S", SPEC_SILENCE_S) / FRAME_S))
    tail = rms[-n:]
    return len(tail) >= n and bool((tail < thr).all())


def _speech_after(rms, thr, start_byte: int) -> bool:
    start_frame = max(0, start_byte // int(FRAME_S * SR * 2))
    seg = rms[start_frame:]
    return bool(seg.size and (seg >= thr).any())


def _maybe_speculate(audio: bytes) -> None:
    """Called on every partial tick. Arms one speculative decode per stretch of
    trailing silence that contains speech the last speculation didn't cover."""
    global _spec_thread, _spec_started
    if not _speculation_enabled() or _finalizing or _spec_alive() or _fc_busy:
        return
    if len(audio) < int(_env_f("DHWANI_SPEC_MIN_AUDIO_S", SPEC_MIN_AUDIO_S) * BYTES_PER_SEC):
        return
    rms = _frame_rms(audio)
    if rms is None:
        return
    thr = _silence_threshold(rms)
    if float(rms.max()) <= 60.0:
        return  # nothing but silence so far — nothing to transcribe
    # PERIODIC speculation (DHWANI_SPEC_PERIODIC_S, off by default).
    #
    # Today a speculation is armed only by trailing silence, so a speaker who
    # never pauses gets none at all and their final is a cold whole-buffer
    # decode. Arming every N seconds regardless covers that case, and pairs with
    # DHWANI_SPEC_JOIN below: the newest COMPLETED speculation then becomes a
    # prefix the final does not have to re-decode.
    #
    # This is the chunked final's job done without the chunked final's cost.
    # DHWANI_CHUNK_S=6 measured -6.90/70 because each committed window is a
    # SEGMENT decode that loses cross-boundary context; every speculation is a
    # WHOLE-BUFFER decode with full context, so the same latency lever is pulled
    # with the quality mechanism that caused the loss removed.
    periodic = _env_f("DHWANI_SPEC_PERIODIC_S", 0.0)
    due = periodic > 0 and (len(audio) - _spec_started) >= int(periodic * BYTES_PER_SEC)
    if not _tail_is_silent(rms, thr) and not due:
        return
    if not _speech_after(rms, thr, _spec_covered):
        return  # the completed speculation already covers all speech

    gen = _clip_gen

    def _run(buf: bytes = audio, g: int = gen) -> None:
        global _spec_text, _spec_covered
        try:
            text = _final_decode(buf)
        except Exception as exc:
            _log_error("speculative final decode (discarded, final will re-decode)", exc)
            return
        with _state_lock:
            if g == _clip_gen:
                _spec_text = text
                _spec_covered = len(buf)

    _spec_started = len(audio)
    t = threading.Thread(target=_run, daemon=True)
    _spec_thread = t
    t.start()


def _spec_take(audio: bytes, deadline: float | None = None) -> str | None:
    """Return the speculative final iff it covers every bit of speech in the
    clip's full audio; None means the caller must decode fresh."""
    import time as _time

    if not _speculation_enabled():
        return None
    rms = _frame_rms(audio)
    if rms is None:
        return None
    thr = _silence_threshold(rms)

    t = _spec_thread
    if t is not None and t.is_alive():
        # Only wait out an in-flight decode whose buffer still covers all
        # speech; if speech resumed after it started, its answer is stale and
        # the fresh decode would have to run anyway.
        if _speech_after(rms, thr, _spec_started):
            return None
        # Wait only within the caller's budget. This used to join for 30s,
        # which on a slow speculation was itself the late final.
        budget = 30.0 if deadline is None else max(0.0, deadline - _time.monotonic())
        t.join(timeout=budget)
        if t.is_alive():
            return None

    with _state_lock:
        text, covered = _spec_text, _spec_covered
    if text is None or not text.strip():
        return None
    if covered > len(audio):
        return None
    if _speech_after(rms, thr, covered):
        return None  # speech landed after the speculation began
    return text


def _spec_prefix() -> tuple[str, int]:
    """The newest COMPLETED speculation, whether or not it covers the clip.

    _spec_take() is all-or-nothing: the moment speech resumes after a
    speculation started, its whole answer is discarded and the final re-decodes
    the buffer from zero — throwing away a full-quality decode of everything
    before that point. Returns ("", 0) when there is nothing.
    """
    with _state_lock:
        return (_spec_text or ""), _spec_covered


def _spec_join_enabled() -> bool:
    return _env_flag("DHWANI_SPEC_JOIN", False)


def _start_spec_join(audio: bytes, deadline: float | None) -> tuple[object, dict] | None:
    """Decode ONLY the audio a completed speculation didn't cover, and join.

    The speculation is a whole-buffer decode of audio[:covered]; the tail is
    everything after. That is exactly the chunked final's join, with two
    properties the committer cannot offer:

      * the split is guaranteed to sit in SILENCE. A speculation only arms when
        the tail of the buffer has gone quiet, so `covered` is a pause by
        construction — the cross-boundary context loss that cost chunking
        -6.90/70 has no word to cut through here.
      * the prefix was decoded with FULL context, not as an isolated ~6s window.

    Returns None when there is nothing better than the ordinary path: no
    completed speculation, it covers everything (_spec_take already took it), or
    the committed windows already reach further.
    """
    if not _spec_join_enabled():
        return None
    text, covered = _spec_prefix()
    if not text.strip() or covered <= 0 or covered >= len(audio):
        return None
    with _state_lock:
        fc_bytes = _fc_bytes
        lang_hint = _fc_lang if _fc_lang_pin() else None
    if covered <= fc_bytes:
        return None      # _final_decode already starts from the further point
    if len(audio) - covered < int(0.2 * BYTES_PER_SEC):
        return None      # nothing but a sliver left; the ordinary path is fine

    box: dict[str, str] = {}

    def _run() -> None:
        try:
            seg_text, _ = _decode_final_segment(
                audio[covered:], pinned_lang=lang_hint,
                prompt=text[-PROMPT_CHARS:], fast=True, deadline=deadline)
            joined = _join_final(text, seg_text) if seg_text.strip() else text
            box["text"] = _normalize_numbers(_deloop(joined))
        except Exception as exc:
            _log_error("speculation prefix-join (falling back to a full tail decode)", exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t, box


# --- chunked final --------------------------------------------------------
#
# A whole-buffer final decode costs O(clip length): fine for a 10s dictation,
# but on a 60s clip it blows past the ~2s latency target no matter when it
# starts (speculation only buys back the trailing silence, a fraction of a long
# clip). So on clips longer than DHWANI_CHUNK_S we close finished windows during
# the clip and lock their text; the end-of-clip decode then only faces the last,
# bounded window. Each window is large (~CHUNK_S, near whisper's 30s context),
# so it decodes as reliably as a batch pass — unlike the 1-3s LocalAgreement
# windows whose thin context caused the hallucination the whole-buffer final was
# built to avoid. Windows close at the quietest frame near the boundary so a
# word is rarely split, and each carries the previous window's tail as a prompt.

def _snap_end(audio: bytes, ideal_end: int, search_s: float = 1.5) -> int:
    """Move a window boundary to the quietest frame within +/- search_s of the
    ideal end, so windows split on pauses rather than through words."""
    rms = _frame_rms(audio)
    fb = int(FRAME_S * SR) * 2   # bytes per 20ms frame
    if rms is None or fb == 0:
        return _even(ideal_end)
    ideal_f = ideal_end // fb
    span = int(search_s / FRAME_S)
    lo, hi = max(0, ideal_f - span), min(len(rms), ideal_f + span)
    if lo >= hi:
        return _even(ideal_end)
    import numpy as np
    best = lo + int(np.argmin(rms[lo:hi]))
    return _even(best * fb)


def _pause_end(audio: bytes, start: int, min_bytes: int) -> int | None:
    """Byte offset of the most recent phrase boundary at least `min_bytes` past
    `start`: a run of PAUSE_S silence with CHUNK_SETTLE_S of audio after it.
    Committing at a pause keeps the uncommitted tail short — which is what the
    end-of-clip decode has to chew through — without ever splitting a word."""
    rms = _frame_rms(audio)
    if rms is None:
        return None
    thr = _silence_threshold(rms)
    fb = int(FRAME_S * SR) * 2
    need = max(1, int(_env_f("DHWANI_PAUSE_S", PAUSE_S) / FRAME_S))
    lo = (start + min_bytes) // fb
    hi = len(rms) - int(_env_f("DHWANI_SETTLE_S", CHUNK_SETTLE_S) / FRAME_S)
    if hi - lo < need:
        return None
    quiet = 0
    for f in range(hi - 1, lo - 1, -1):     # newest boundary first
        if rms[f] < thr:
            quiet += 1
            if quiet >= need:
                return _even((f + need // 2) * fb)
        else:
            quiet = 0
    return None


def _maybe_commit_window(audio: bytes) -> None:
    """Close and lock the next finished window in the background.

    A window closes either at the newest phrase boundary past COMMIT_MIN_S
    (preferred — keeps the tail short and cuts on a pause), or, if the speaker
    never pauses, at the hard CHUNK_S boundary.
    """
    global _fc_thread, _fc_busy
    chunk = _chunk_s()
    if chunk <= 0 or _fc_busy or _spec_alive() or _finalizing:
        return
    window_bytes = _even(int(chunk * BYTES_PER_SEC))
    settle = int(_env_f("DHWANI_SETTLE_S", CHUNK_SETTLE_S) * BYTES_PER_SEC)
    with _state_lock:
        start = _fc_bytes
    pause_end = _pause_end(audio, start,
                           _even(int(_env_f("DHWANI_COMMIT_MIN_S", COMMIT_MIN_S) * BYTES_PER_SEC)))
    if pause_end is None and len(audio) - start < window_bytes + settle:
        return

    _fc_busy = True
    gen = _clip_gen

    def _run(buf: bytes = audio, s: int = start, g: int = gen,
             pe: int | None = pause_end) -> None:
        global _fc_busy, _fc_text, _fc_bytes, _fc_lang
        try:
            with _state_lock:
                lang, prompt = (_fc_lang if _fc_lang_pin() else None), _fc_text[-PROMPT_CHARS:]
            end = pe if pe is not None else _snap_end(buf, s + window_bytes)
            if end <= s + int(0.5 * BYTES_PER_SEC):
                end = _even(s + window_bytes)   # snap found nothing usable
            end = min(end, len(buf))
            seg_text, detected = _decode_final_segment(
                buf[s:end], pinned_lang=lang, prompt=prompt)
            with _state_lock:
                if g == _clip_gen and _fc_bytes == s:
                    _fc_text = _join_final(_fc_text, seg_text)
                    _fc_bytes = end
                    if _fc_lang is None and _fc_lang_pin():
                        _fc_lang = detected
        except Exception as exc:
            _log_error("window committer decode", exc)
        finally:
            _fc_busy = False

    t = threading.Thread(target=_run, daemon=True)
    _fc_thread = t
    t.start()


def _fc_alive() -> bool:
    t = _fc_thread
    return t is not None and t.is_alive()


# --- helpers --------------------------------------------------------------

def map_words_triples(words, lang):
    """Run the orthography adapter at word level.

    Word level matters: the adapter must transform a word identically whether it
    lands in a partial or in the final, or the committed prefix would disagree
    with the final and the scorer would charge us revision churn.
    """
    if not words:
        return []
    mapped = map_words([t for (t, _, _) in words], lang)
    return [(m, s, e) for (m, (_, s, e)) in zip(mapped, words) if m]


def _norm(word: str) -> str:
    return _PUNCT.sub("", word).lower()


def _common_prefix(a, b) -> int:
    n = 0
    for (wa, _, _), (wb, _, _) in zip(a, b):
        if _norm(wa) != _norm(wb):
            break
        n += 1
    return n


def _even(n: int) -> int:
    return (n // 2) * 2


def _break_repetition_loop(text: str, max_n: int = 4, k: int = 4) -> str:
    """Collapse an n-gram repeated k+ times in a row down to one copy.

    The scorecard caps a repetition loop at 30. Whisper falls into them on silence
    and on unfamiliar phonetics: a Hinglish clip here decoded as the single token
    'अद्राद' seven times over. Checking only 3-grams misses that, so scan n=1..4.
    """
    toks = text.split()
    if not toks:
        return text

    out: list[str] = []
    i = 0
    while i < len(toks):
        for n in range(1, max_n + 1):
            gram = [t.lower() for t in toks[i:i + n]]
            if len(gram) < n:
                continue
            j, reps = i + n, 1
            while [t.lower() for t in toks[j:j + n]] == gram:
                reps, j = reps + 1, j + n
            if reps >= k:
                out.extend(toks[i:i + n])
                i = j
                break
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


# a unit of 1..16 chars repeated 4+ times in a row, collapsed to one copy —
# catches the space-less Devanagari loops ("इस प्रशाइशाइशाइशाइ...") that the
# whitespace tokenizer above cannot see.
_CHAR_LOOP = re.compile(r"(.{1,16}?)\1{3,}", flags=re.UNICODE)
_ARABIC = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")


def _deloop(text: str) -> str:
    return _break_repetition_loop(_CHAR_LOOP.sub(r"\1", text or "")).strip()


_DEV_DIGITS_T = str.maketrans("०१२३४५६७८९", "0123456789")
_VERSIONISH = re.compile(r"\b\d+(?:\.\d+){2,}\b")   # 2+ dots: 3.3.4, not 3.5
_DIGIT_HYPHEN = re.compile(r"(?<=\d)-(?=\d)")

# Spoken numbers, English and Hindi (Devanagari + common romanisations). The
# scorer pulls required numbers out of the GOLD and demands each appears
# verbatim in the final, so a gold "100 साल" against a transcribed "सो साल" is
# a fact flip that caps the clip at 50 — measured on fleurs_hi_in_test_1718.
_NUM_WORDS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
    "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
    "eighty": 80, "ninety": 90,
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5,
    "छह": 6, "छः": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10, "ग्यारह": 11,
    "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "सोलह": 16, "सत्रह": 17,
    "अठारह": 18, "उन्नीस": 19, "बीस": 20, "तीस": 30, "चालीस": 40, "पचास": 50,
    "साठ": 60, "सत्तर": 70, "अस्सी": 80, "नब्बे": 90,
    "ek": 1, "do": 2, "teen": 3, "chaar": 4, "paanch": 5,
    "saat": 7, "aath": 8, "nau": 9, "das": 10, "bees": 20, "pachaas": 50,
}
_NUM_SCALES: dict[str, int] = {
    "hundred": 100, "thousand": 1000, "million": 1000000,
    "सौ": 100, "सो": 100, "हज़ार": 1000, "हजार": 1000, "लाख": 100000,
    "करोड़": 10000000, "sau": 100, "hazaar": 1000, "hajaar": 1000, "lakh": 100000,
}
_NUM_JOIN = {"and", "aur", "और"}


def _words_to_number(tokens: list[str]) -> int | None:
    """Value of a run of number words, or None if it isn't a clean number."""
    total = current = 0
    seen = False
    for tok in tokens:
        low = tok.lower()
        if low in _NUM_WORDS:
            current += _NUM_WORDS[low]
            seen = True
        elif low in _NUM_SCALES:
            scale = _NUM_SCALES[low]
            current = (current or 1) * scale
            if scale >= 1000:
                total += current
                current = 0
            seen = True
        elif low in _NUM_JOIN:
            continue
        else:
            return None
    return (total + current) if seen else None


def _digitize_spoken_numbers(text: str) -> str:
    """Rewrite spoken numbers as digits so they can match the gold.

    Deliberately conservative: a run is only rewritten when it spans more than
    one number word or reaches 20. Single small words are left alone because
    "एक"/"one" is far more often the article "a" than the figure 1, and
    rewriting those would cost meaning tokens for no factual gain.
    """
    if not text:
        return text
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if not run:
            return
        core = [t for t in run if t.lower() not in _NUM_JOIN]
        value = _words_to_number(run)
        if value is not None and (len(core) > 1 or value >= 20):
            out.append(str(value))
        else:
            out.extend(run)
        run.clear()

    for token in text.split():
        bare = token.strip(".,!?;:()\"'")
        low = bare.lower()
        if low in _NUM_WORDS or low in _NUM_SCALES or (run and low in _NUM_JOIN):
            run.append(bare)
            continue
        flush()
        out.append(token)
    flush()
    return " ".join(out)


def _normalize_numbers(text: str) -> str:
    """Final-only number cleanup, driven by two real losses on the Mac run.

    The scorer's critical_flip() extracts number tokens from RAW text with
    \\b\\d[\\d,.:/-]*\\b and requires every gold number to appear verbatim —
    one mismatch zeroes the 20-point facts axis and caps the clip at 50.

    * Whisper heard "version 334" and wrote "3.3.4" — the corpus writes bare
      digits ("334"), so collapse dotted runs with 2+ dots. Decimals like
      "3.5" (one dot) are deliberately untouched.
    * Whisper wrote "25-30" where the gold says "25 to 30"; the raw-text regex
      reads "25-30" as ONE number, matching neither 25 nor 30. Splitting the
      hyphen yields both. (A gold that itself hyphenates would prefer the
      joined form; this corpus doesn't in any manifest we can see.)
    * Devanagari digits become ASCII so "३३४" can match gold "334".
    """
    if not text:
        return text
    text = text.translate(_DEV_DIGITS_T)
    text = _digitize_spoken_numbers(text)
    text = _VERSIONISH.sub(lambda m: m.group(0).replace(".", ""), text)
    return _DIGIT_HYPHEN.sub(" ", text)


_LATIN = re.compile(r"[A-Za-z]")
_NUMBER_RX = re.compile(r"\b\d[\d,.:/-]*\b")   # the scorer's own number extractor


def _has_latin(text: str) -> bool:
    return bool(_LATIN.search(text or ""))


def _drops_numbers(primary: str, candidate: str) -> bool:
    """True if the primary transcript contains a multi-digit number token the
    candidate lost. Both sides are number-normalized first, mirroring what the
    scorer will see. Single digits are ignored — too noisy to arbitrate on."""
    prim = {n.replace(",", "") for n in _NUMBER_RX.findall(_normalize_numbers(primary or ""))}
    prim = {n for n in prim if len(re.sub(r"\D", "", n)) >= 2}
    if not prim:
        return False
    cand = {n.replace(",", "") for n in _NUMBER_RX.findall(_normalize_numbers(candidate or ""))}
    return not prim.issubset(cand)


def _looks_bad(text: str) -> bool:
    """A final that should trigger a Hindi re-decode: empty, still looping after
    _deloop, or written in Arabic script (an Urdu mis-detect that normalizes to
    nothing on the scorecard)."""
    if not text or not text.strip():
        return True
    if _deloop(text) != text.strip():
        return True
    if _ARABIC.search(text):
        return True
    return False


# --- cold start -----------------------------------------------------------
#
# The sealed stream_server prints `READY port=...` as soon as the socket is
# listening; it never calls warm_models(). So the model load lands lazily on
# the FIRST draft() call, i.e. inside the first scored clip. Measured on the
# local streaming harness: clip 1 took 23016 ms end-to-final and returned the
# best-effort text, while later clips on the same run finished in 731-1355 ms.
#
# Warming runs on a BACKGROUND thread, deliberately. Doing it inline would move
# the load in front of `READY`, and evaluator.py terminates a server that has
# not printed READY within 60 seconds — turning a slow first clip into a failed
# run. This way READY is immediate and the load overlaps the harness's own
# setup. Guarded so the test suite and the offline tooling, which import this
# module constantly, never trigger a model download.

def _warm_on_import() -> bool:
    """Unset — the scored default — means auto-detect: warm only inside the
    sealed server process. "1" forces it on so the local streaming harness can
    reproduce the server's cold start; "0" forces it off for the test suite."""
    forced = os.environ.get("DHWANI_WARM_ON_IMPORT")
    if forced is not None:
        return forced != "0"
    import sys
    if "solution.stream_server" in sys.modules:
        return True
    return os.path.basename(sys.argv[0] or "") == "stream_server.py"


if _warm_on_import():
    try:
        threading.Thread(target=warm_models, daemon=True,
                         name="dhwani-warm").start()
    except Exception as exc:      # never let warming stop the server starting
        _log_error("warm-on-import (models will load on the first clip)", exc)
