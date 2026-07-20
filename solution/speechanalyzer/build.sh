#!/usr/bin/env bash
# Compile the SpeechAnalyzer CLI helper. Needs macOS 26 (Tahoe) + Xcode 26.
#
# Preflight-checks the OS and toolchain FIRST, because without the macOS 26 SDK
# the compiler emits ~20 "cannot find type 'SpeechTranscriber' in scope" errors
# that look like broken code but actually just mean "wrong SDK". One clear
# diagnosis beats a wall of red.
set -e

HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$HERE/speechanalyzer/speechanalyzer_cli.swift"
OUT="$HERE/speechanalyzer/speechanalyzer_cli"

if [ "$(uname)" != "Darwin" ]; then
    echo "SKIP: SpeechAnalyzer is Apple-only. Needs macOS 26 (Tahoe) on Apple silicon."
    exit 1
fi

MACOS="$(sw_vers -productVersion 2>/dev/null || echo "unknown")"
MAJOR="${MACOS%%.*}"
echo "macOS:  $MACOS"

if ! command -v swiftc >/dev/null 2>&1; then
    echo "swiftc not found. Install Xcode 26 from the App Store, then:  xcode-select --install"
    exit 1
fi

SWIFTV="$(swiftc --version 2>/dev/null | head -1)"
echo "swift:  $SWIFTV"
echo "sdk:    $(xcrun --show-sdk-version 2>/dev/null || echo unknown) (via $(xcode-select -p 2>/dev/null))"
echo

# Swift 6.x ships with Xcode 26; Swift 5.x means an Xcode 15/16-era toolchain
# whose SDK has no SpeechAnalyzer at all.
SWIFT_MAJOR="$(echo "$SWIFTV" | sed -n 's/.*version \([0-9]*\)\..*/\1/p')"

if [ "$MAJOR" != "unknown" ] && [ "${MAJOR:-0}" -lt 26 ] 2>/dev/null; then
    echo "CANNOT BUILD: this Mac runs macOS $MACOS."
    echo "Apple's SpeechAnalyzer API only exists on macOS 26 (Tahoe) and later —"
    echo "it is not a library that can be installed, it ships with the OS."
    echo "Nothing to do here unless you upgrade macOS. The Whisper backends are"
    echo "unaffected and remain the default."
    exit 1
fi

if [ -n "$SWIFT_MAJOR" ] && [ "$SWIFT_MAJOR" -lt 6 ] 2>/dev/null; then
    echo "CANNOT BUILD: Swift $SWIFT_MAJOR toolchain (Xcode 15/16 era)."
    echo "SpeechAnalyzer needs the macOS 26 SDK, which ships with Xcode 26 (Swift 6+)."
    echo "If Xcode 26 is installed, point the toolchain at it:"
    echo "    sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
    echo "    swiftc --version     # expect Swift 6.x"
    echo "Otherwise install Xcode 26 from the App Store. (Command Line Tools alone"
    echo "are often the culprit — they can lag the OS by a full release.)"
    exit 1
fi

echo "compiling $SRC ..."
swiftc -O "$SRC" -o "$OUT"
chmod +x "$OUT"
echo "built: $OUT"
echo
echo "quick manual check (compare against samples/manifest.json gold):"
echo "  $OUT samples/openslr104_hi_en_103085_w5Jyq3XMbb3WwiKQ_0000.wav hi"
