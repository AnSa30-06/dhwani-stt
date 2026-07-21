# builderr · Dual-Language Speech-to-Text Challenge

> ## This fork: the **dhwani** entry (streaming track)
>
> Everything of ours lives in `solution/draft.py` (+ `solution/orthography.py`,
> `solution/speechanalyzer/`); the harness files are untouched. Design in one
> line: **the scored final is a fresh whole-buffer decode that starts
> speculatively during the speaker's trailing silence** (≈full latency points,
> zero quality risk), with a **language router** — English/default finals on
> `large-v3-turbo` (mlx on Apple silicon), Hindi/code-switched finals
> re-decoded on `shunyalabs/zero-stt-hinglish` via transformers (MPS on the
> scoring box), better candidate kept, with a number-preservation guard.
>
> **Before official (offline) scoring, pre-cache the models once, online:**
>
> ```bash
> pip install -r requirements.txt -r requirements-streaming.txt
> python -c "from solution.draft import warm_models; warm_models()"
> ```
>
> The official server runs with `HF_HUB_OFFLINE=1` inside sandbox-exec, so
> models must already be in the HF cache; `warm_models()` fetches exactly the
> configured set (~3.5 GB). Zero-config defaults ARE the submission
> configuration; env knobs are documented at the top of `solution/draft.py`.
> Solution tests: `python -m pytest tests/test_dhwani.py`.
>
> Models + licenses: OpenAI whisper `large-v3-turbo` + `small` (MIT, via
> mlx-community / Systran conversions) · `shunyalabs/zero-stt-hinglish`
> (OpenRAIL). Measured on 15 local clips (quality axis, scored with this
> repo's own scorer): zero-stt **58.8/70 on Hinglish** vs turbo's 37.4, tie on
> pure Hindi (51.4 / 51.3); English stays on turbo (53.3).

**Build a speech-to-text tool that runs on your own laptop, works offline, and actually
understands when people mix Hindi and English.** Win **$500** — and walk away with a
project genuinely worth putting on GitHub.

**Start here:** [**Getting started**](GETTING_STARTED.md) — your first transcriber in a few steps ·
[**Build skill**](SKILL.md) — the high-level architecture to follow (drop it into your AI) ·
[**Sample clips**](samples/) — real English + Hindi+English clips to test on ·
[**Reference bot**](docs/REFERENCE_BOT.md) — the bar to beat, and how it's built.

## The problem (and why it's worth building)

If you build with AI, you talk to it all day — and typing is the slow part. Speech-to-text
fixes that. But two things get in the way:

- The good tools are **cloud tools** (like Wispr Flow). They **cost money**, and lots of
  companies **won't let you use them** — your words leave your computer, and that's a
  no-go for them.
- And most tools **fall apart the moment you mix languages.** Tons of us don't speak in
  just one — we slide between **Hindi and English in the same sentence.** Cloud tools
  mangle that, and local ones are worse.

So here's the gold: **a speech-to-text tool that runs locally (no internet, nothing leaves
your machine), is fast enough that talking beats typing, and gets mixed Hindi+English
right — the actual words *and* what you meant.**

Nail that and it's useful to a huge number of people. **Win this challenge with it — then
put it on GitHub.** A free, private, mixed-language dictation tool that just works? That
pulls real stars. It's a great thing to have built, whether you win or not.

## What it takes to win — in plain words

Your tool has to:

1. **Get plain English right** — about as accurate as the best free tools out there.
   (You don't have to *beat* them on English; tying is fine.)
2. **Be fast** — for quick English it should feel almost instant (**under ~1 second**),
   so talking is faster than typing.
3. **Nail Hindi+English mixing** — *this is the real test.* When someone mixes languages,
   write down **what they actually said** (don't quietly translate it all into English),
   and **keep the meaning.** This is exactly where today's tools break — and where you win.
4. **Stay on the laptop** — no internet while it's scored. Everything runs locally.
5. **Be shippable** — runs on a normal computer (not a giant model), works on **Mac and
   Linux**, and uses only models that are **free to use in your own product.** So you can
   actually release it.
6. **Not cheat or break** — no hard-coded answers, no crashes, no repeating-gibberish loops.

**This isn't meant to be easy — that's the point.** Just wrapping an off-the-shelf model gets
you good English and broken Hindi+English. **You can't win on English alone** — the mixed-language
part is the gate, and that's where the real building is (a router + a Hindi-capable model + a
finalizer that stays faithful and fast, all local). And to take the $500, **your tool has to
actually beat the best thing out there today** — the top free tools *and* our own engine — on
the hidden set. If nothing clearly beats it, we award nothing. We're paying for a real step up,
not a tie.

<details><summary><b>The exact bar</b> (for the technically inclined)</summary>

| What | Target | Where things are today |
| --- | --- | --- |
| English accuracy | word-error ≤ ~0.06, meaning ≥ 0.95 | best free engine ~0.06; you're already here |
| English speed | fast-mode p95 < 0.8s (warm) | best free engine ~0.45s; the real lift |
| Hindi+English meaning | ≥ 0.90 | today's best local ~0.84 |
| Hindi+English faithfulness | keeps the code-switch — word-error ≤ ~0.25 | cloud/free tools ~0.91 (they translate it away) |
| Reliability | ~0 blanks, <1% hangs, no loops | — |
| Footprint / license | ≤ ~5GB, CPU-runnable, commercial-friendly models | — |

Final numbers lock after the full hidden-set baseline run; these are the validated targets
from a live head-to-head against the best open-source engines.
</details>

## The contract you implement

```bash
python -m solution.transcribe --input clip.wav --mode auto --output result.json
```

Emit the JSON in [`solution/transcribe.py`](solution/transcribe.py) (`text`, `mode_used`,
`language_guess`, `timings_ms`, `raw_candidates`, `model_ids`, `local_only`).
Modes: `auto` / `fast` / `hinglish` / `verbatim`.

## The streaming dictation track — where the $500 is

There's one prize, **$500**, on one combined score, decided on **live dictation**:
the final text must preserve what the speaker meant and arrive quickly after they
stop. You write **one function** — `draft()` in
[`solution/draft.py`](solution/draft.py). The streaming server and real-time audio feed are a
**sealed harness we provide** ([`solution/stream_server.py`](solution/stream_server.py)) —
you do not build a server.

You're scored on one combined number: **final transcript correctness (70 points)**
and **time from stopping to the final paste (30 points)**. Intermediate drafts are
optional and never affect the score.
**RambleFix is the benchmark line to beat — it can't win the prize.**

The full contract, scoring, caps, frozen-CPU bar, and RambleFix's published numbers
are the single source of truth in
[**`docs/STREAMING_CONTRACT.md`**](docs/STREAMING_CONTRACT.md). Preview yourself,
offline, exactly like admission:

```bash
pip install -r requirements.txt -r requirements-streaming.txt
python preview_stream.py
```

## Run the local preview (scores you exactly like admission, offline)

```bash
pip install -r requirements.txt   # a local ASR engine (faster-whisper, whisper.cpp, parakeet…)
python preview.py
```

## How you're scored (`scorecard.py`, out of 100)

| Area | Weight | In plain words |
| --- | ---: | --- |
| Meaning accuracy | 40 | did it keep what you meant? |
| Critical facts & terms | 25 | dates, numbers, "not", acronyms, names — a flip caps the clip |
| Latency & reliability | 20 | fast, no blanks, no hangs |
| Local-only proof | 10 | ran with the internet off |
| Auditability | 5 | shows its working (candidates, timings, model IDs) |

**Hard caps:** a critical flip caps a clip at 50; a blank at 20; a repetition loop at 30;
output unrelated to the audio at 20. **Any network call during scoring fails. Hard-coded
phrases / hidden answer maps fail.**

## Rules

- Fully local; the scored run blocks outbound internet after warmup (loopback to a local
  ASR server is fine — see `offline_guard.py`).
- **Declare your models and their licenses — they must be commercial-friendly** (so the
  winning tool can actually be released for free).
- p95 under 5s, hang/error under 1%, no hard-coded phrase fixes.

## Submit

Email your repo to **submit@builderr.ai**. We clone it, run it offline on the hidden set
on a Linux box, and you land on the board.

## Tests

```bash
python tests/test_scorecard.py            # batch scoring is fair + un-gameable
python tests/test_no_network.py           # offline enforcement works
pytest tests/test_streaming_scorecard.py  # final transcript + final latency scoring
pytest tests/test_stream_contract.py      # the sealed streaming server speaks the contract
```
