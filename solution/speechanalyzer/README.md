# SpeechAnalyzer backend (experimental, Apple silicon + macOS 26 only)

`speechanalyzer_cli.swift` wraps Apple's on-device **SpeechAnalyzer** API so
dhwani can use Apple's speech model instead of Whisper for transcription.

## Why this might be worth it

Apple's SpeechAnalyzer (new in macOS 26 Tahoe) benchmarks *better than Whisper
Small* on English — 2.12% vs 3.74% WER on LibriSpeech — and runs ~3× faster,
fully on-device. The challenge's frozen scoring box runs macOS Tahoe 26.3.1, so
this API is available there. If it also handles the Hindi+English clips well,
it could beat the Whisper backend on both the quality and latency axes.

## The open question this backend exists to answer

**Nobody knows yet whether SpeechAnalyzer produces good output on the challenge's
Hindi clips.** Megaphone — the app this is adapted from — routes "Hinglish"
through Apple's *Indian-English* model (`en-IN`) and does Devanagari handling in
a separate LLM step, which suggests SpeechAnalyzer alone may not cleanly produce
the mixed Devanagari+Latin script the challenge gold uses. It might do well with
`locale=hi`; it might romanize; it might miss English terms. The only way to know
is to run it on the actual clips — which is what `tools/speechanalyzer_eval.py`
does in one command.

**Measure first, integrate second.** Don't switch dhwani to this backend until
the eval shows it beats Whisper's 41.3/70 on the sample clips.

## Build

```bash
cd speechanalyzer
swiftc -O speechanalyzer_cli.swift -o speechanalyzer_cli
```

Requires the macOS 26 SDK (Xcode 26). `tools/build_speechanalyzer.sh` does this
from the project root and checks for the toolchain first.

## Try it directly (the fastest way to see if it's any good)

```bash
# one Hindi clip, Hindi model
./speechanalyzer_cli ../samples/openslr104_hi_en_103085_w5Jyq3XMbb3WwiKQ_0000.wav hi
# same clip, Indian-English model (Megaphone's Hinglish choice)
./speechanalyzer_cli ../samples/openslr104_hi_en_103085_w5Jyq3XMbb3WwiKQ_0000.wav en-IN
```

Compare what each prints against the gold in `../samples/manifest.json`.

## Credit

Adapted from **[Megaphone](https://github.com/Kuberwastaken/megaphone)** by
Kuber Mehta (MIT), specifically `Sources/SpeechAnalyzerService.swift` — its
file-based `transcribe(fileURL:)` path and `SpeechLocaleResolver`. Megaphone is a
fork of **[FreeFlow](https://github.com/zachlatta/freeflow)** by Zach Latta and
contributors. Both are MIT licensed; see `LICENSE-megaphone` in this folder. This
wrapper adds only the CLI/serve plumbing so a non-Swift process can call it.
