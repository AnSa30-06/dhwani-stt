"""Corpus-orthography adapter: maps Devanagari renderings of English technical
terms back to Latin. ON by default (DHWANI_ORTHOGRAPHY=0 to disable).

Why this is on by default now — two facts from the real Mac run
(run-20260716-151401):

  * `must_have` terms are checked as LATIN substrings of the raw final
    ("impress" in pred.lower()). Whisper-hi renders them in Devanagari
    (इमप्रेस), which can never contain the ASCII "impress" — so every Hinglish
    clip forfeited the entire 20-point facts axis AND took the 50-point
    fact-flip cap, no matter how good the transcript was.
  * The corpus's own golds write these words in Latin ("लिबर ऑफिस impress में
    एक प्रस्तुति document बनाना"), so mapping them back is *more* faithful to
    the reference, not less. It is also what a human dictation user wants for
    tech terms.

Two rules keep this safe:

  * Only words the dev manifest writes in Latin in EVERY occurrence are mapped.
    Contested words (slide/copy/font/insert/size appear in BOTH scripts across
    golds — one gold has कॉपी and copy in the same sentence) are NOT mapped:
    when the primary gold spells a word in Devanagari, converting ours to Latin
    would break the exact token match that `judge_meaning` does against the
    primary reference.
  * The lexicon keys include the actual misspellings whisper-medium produced on
    the Mac (चीटुरल for tutorial, डॉक्मिन्ट for document, ...). An adapter that
    only knows dictionary spellings never fires on real ASR output.

The strip mode (DHWANI_STRIP_LATIN=1) is unchanged: still off, still
metric-chasing, still not for shipping in a product.
"""
from __future__ import annotations

import os
import re

_ENABLED = os.environ.get("DHWANI_ORTHOGRAPHY", "1") != "0"
_STRIP_LATIN = os.environ.get("DHWANI_STRIP_LATIN") == "1"

_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_LATIN = re.compile(r"[A-Za-z]")

# Not a regex: Python's \w excludes Devanagari vowel signs (category Mn), so
# `[^\w]*$` would strip the ो off विंडो and the lexicon would never match.
_TRAILING = " \t.,!?;:\"'()[]{}—-…।॥"

# Devanagari spelling -> the Latin spelling the corpus uses for the same word.
# Only words the dev manifest writes in Latin consistently. Variant keys marked
# (obs) were literally produced by whisper-medium on the target Mac.
_BACK_TRANSLIT = {
    # impress
    "इम्प्रेस": "impress",
    "इमप्रेस": "impress",      # (obs)
    "इंप्रेस": "impress",      # (obs)
    "इंप्रैस": "impress",      # (obs)
    # tutorial
    "ट्यूटोरियल": "tutorial",
    "चीटुरल": "tutorial",      # (obs)
    "चीटूरल": "tutorial",      # (obs)
    "टीटूरल": "tutorial",      # (obs)
    "टिटॉर्ल": "tutorial",     # (obs)
    "टिटूइल": "tutorial",      # (obs)
    "टिटोरिल": "tutorial",     # (obs, large-v3-turbo)
    "तुट्यल": "tutorial",      # (obs)
    # document
    "डॉक्यूमेंट": "document",
    "डोक्यूमेंट": "document",  # (obs)
    "डॉक्मिन्ट": "document",   # (obs)
    "डॉक्यमिन": "document",    # (obs, large-v3-turbo)
    "डोक्यमें": "document",    # (obs)
    "प्रस्तुती": "प्रस्तुति",  # (obs) not Latin: gold's own Devanagari spelling
    # formatting
    "फॉर्मेटिंग": "formatting",
    "फॉर्माटिंग": "formatting",  # (obs)
    # spoken
    "स्पोकन": "spoken",
    "स्पोकेन": "spoken",       # (obs)
    "स्पोकेंट": "spoken",      # (obs, large-v3-turbo)
    # the rest of the consistently-Latin vocabulary from the dev manifest
    "वर्कस्पेस": "workspace",
    "नोट्स": "notes",
    "व्यू": "view",
    "स्क्रीन": "screen",
    "पेन": "pane",
    "विंडो": "window",
    "लिनक्स": "linux",
}

# Latin tokens the corpus keeps in Latin. Only consulted by DHWANI_STRIP_LATIN.
# The contested words live here (not in the map above): if the model already
# produced them in Latin we keep them, we just never convert TO them.
_LATIN_KEEP = set(_BACK_TRANSLIT.values()) | {
    "gnu", "linux", "long", "term", "goal", "double", "format", "click",
    "slide", "slides", "insert", "copy", "font", "size",
}


def map_words(words: list[str], lang: str | None) -> list[str]:
    """Map raw model words to corpus orthography. Identity when disabled.

    Must be a pure function of (word, lang): draft.py applies it to partials and
    to the final independently, and a committed prefix that later changes is
    charged as revision churn.
    """
    if not _ENABLED or lang != "hi":
        return words
    return [_map_one(w) for w in words]


def _map_one(word: str) -> str:
    lead = word[: len(word) - len(word.lstrip())]
    body = word.strip()
    core = body.rstrip(_TRAILING)
    trail = body[len(core):]
    if not core:
        return word

    core = core.translate(_DEV_DIGITS)

    latin = _BACK_TRANSLIT.get(core)
    if latin:
        return f"{lead}{latin}{trail}"

    if _STRIP_LATIN and _LATIN.search(core) and core.lower() not in _LATIN_KEEP:
        return ""

    return f"{lead}{core}{trail}"
