// speechanalyzer_cli — a tiny command-line wrapper around Apple's on-device
// SpeechAnalyzer (macOS 26 Tahoe, Apple silicon). Lets dhwani's Python engine
// use Apple's speech model as a transcription backend instead of Whisper.
//
// Adapted from Megaphone (https://github.com/Kuberwastaken/megaphone), MIT
// licensed, specifically Sources/SpeechAnalyzerService.swift's file-based
// `transcribe(fileURL:)` path and its SpeechLocaleResolver. Megaphone is itself
// a fork of FreeFlow (https://github.com/zachlatta/freeflow). Full credit to
// Kuber Mehta and the FreeFlow contributors; see speechanalyzer/README.md.
//
// Build:  swiftc -O speechanalyzer_cli.swift -o speechanalyzer_cli
// Requires: macOS 26 SDK (Xcode 26). Apple silicon at runtime.
//
// Two modes:
//   one-shot:  ./speechanalyzer_cli <wav_path> <locale>
//              prints the transcript to stdout, exits 0. exit != 0 on error,
//              with the reason on stderr.
//   serve:     ./speechanalyzer_cli --serve
//              reads "<wav_path>\t<locale>" lines from stdin, prints one line
//              of transcript per request (newlines in the transcript are
//              escaped to \\n). Loads nothing until the first request; keeps the
//              process warm so per-request cost is just the transcription, not
//              process startup. Empty transcript prints an empty line.
//
// Locale is a BCP-47 code ("hi", "hi-IN", "en-US", "en-IN") or "auto".
// "hinglish"/"gujlish" are accepted and map to en-IN, matching Megaphone.

import AVFoundation
import Foundation
import Speech

// MARK: - Locale resolution (from Megaphone's SpeechLocaleResolver, trimmed)

enum LocaleResolver {
    static let legacyAliases: [String: String] = ["hinglish": "en-IN", "gujlish": "en-IN"]

    static func resolve(_ preference: String) async throws -> Locale {
        let trimmed = preference.trimmingCharacters(in: .whitespacesAndNewlines)
        let supported = await SpeechTranscriber.supportedLocales
        guard !supported.isEmpty else {
            throw Err.msg("SpeechTranscriber reports no supported locales (unavailable on this Mac).")
        }
        if trimmed.isEmpty || trimmed.lowercased() == "auto" {
            return bestMatch(Locale.current, supported)
                ?? supported.first { $0.identifier(.bcp47) == "en-US" }
                ?? supported[0]
        }
        let key = legacyAliases[trimmed.lowercased()] ?? trimmed
        if let match = bestMatch(Locale(identifier: key), supported) { return match }
        let list = supported.map { $0.identifier(.bcp47) }.sorted().joined(separator: ", ")
        throw Err.msg("locale \"\(trimmed)\" not supported. Supported: \(list)")
    }

    static func bestMatch(_ locale: Locale, _ supported: [Locale]) -> Locale? {
        let target = locale.identifier(.bcp47)
        if let exact = supported.first(where: { $0.identifier(.bcp47) == target }) { return exact }
        guard let lang = locale.language.languageCode?.identifier, !lang.isEmpty else { return nil }
        return supported.first { $0.language.languageCode?.identifier == lang }
    }
}

enum Err: Error, CustomStringConvertible {
    case msg(String)
    var description: String { if case let .msg(m) = self { return m } else { return "error" } }
}

// MARK: - Transcription (from Megaphone's SpeechAnalyzerService.transcribe)

func ensureAssets(_ transcriber: SpeechTranscriber, _ locale: Locale) async throws {
    let target = locale.identifier(.bcp47)
    let installed = await SpeechTranscriber.installedLocales
    if !installed.contains(where: { $0.identifier(.bcp47) == target }) {
        if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
            FileHandle.standardError.write("[speechanalyzer] downloading model for \(target)…\n".data(using: .utf8)!)
            try await request.downloadAndInstall()
        }
    }
    let reserved = await AssetInventory.reservedLocales
    if !reserved.contains(where: { $0.identifier(.bcp47) == target }) {
        for stale in reserved { await AssetInventory.release(reservedLocale: stale) }
        try? await AssetInventory.reserve(locale: locale)
    }
}

func transcribeFile(_ path: String, _ localePref: String) async throws -> String {
    guard SpeechTranscriber.isAvailable else {
        throw Err.msg("SpeechTranscriber unavailable (needs macOS 26 on Apple silicon).")
    }
    let locale = try await LocaleResolver.resolve(localePref)
    let transcriber = SpeechTranscriber(locale: locale, transcriptionOptions: [],
                                        reportingOptions: [], attributeOptions: [])
    try await ensureAssets(transcriber, locale)
    let analyzer = SpeechAnalyzer(modules: [transcriber])

    let collector = Task { () -> String in
        var transcript = AttributedString("")
        for try await result in transcriber.results { transcript += result.text }
        return String(transcript.characters)
    }

    let audioFile = try AVAudioFile(forReading: URL(fileURLWithPath: path))
    guard let last = try await analyzer.analyzeSequence(from: audioFile) else {
        collector.cancel(); await analyzer.cancelAndFinishNow()
        throw Err.msg("no audio samples reached the transcriber")
    }
    try await analyzer.finalizeAndFinish(through: last)
    let transcript = try await collector.value
    return transcript.trimmingCharacters(in: .whitespacesAndNewlines)
}

// MARK: - Modes

func runOneShot(_ path: String, _ locale: String) async -> Int32 {
    do {
        let text = try await transcribeFile(path, locale)
        print(text)
        return 0
    } catch {
        FileHandle.standardError.write("[speechanalyzer] ERROR: \(error)\n".data(using: .utf8)!)
        return 1
    }
}

func emit(_ line: String) {
    // stdout to a pipe is block-buffered by default; flush every line so the
    // Python side (which reads line-by-line) never blocks waiting on the buffer.
    print(line)
    fflush(stdout)
}

func runServe() async {
    emit("READY")  // Python blocks on this line before sending the first request
    while let line = readLine(strippingNewline: true) {
        let parts = line.split(separator: "\t", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { emit(""); continue }
        do {
            let text = try await transcribeFile(parts[0], parts[1])
            emit(text.replacingOccurrences(of: "\n", with: "\\n"))
        } catch {
            FileHandle.standardError.write("[speechanalyzer] ERROR: \(error)\n".data(using: .utf8)!)
            emit("")  // empty line = this request failed; caller falls back
        }
    }
}

// MARK: - Entry

let args = CommandLine.arguments
if args.count == 2 && args[1] == "--serve" {
    await runServe()
    exit(0)
} else if args.count == 3 {
    exit(await runOneShot(args[1], args[2]))
} else {
    FileHandle.standardError.write(
        "usage: speechanalyzer_cli <wav_path> <locale>   |   speechanalyzer_cli --serve\n"
            .data(using: .utf8)!)
    exit(2)
}
