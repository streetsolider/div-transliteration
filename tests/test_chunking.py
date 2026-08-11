"""Regression tests for the phrase chunker in app.py.

These exist because a real bug shipped for four months without them: the
stitcher dropped a fixed `overlap` number of words from every non-first chunk,
assuming N overlapped *input* words came back as N *output* words. Dhivehi
transliteration merges words, so whenever the overlap region compressed the trim
ate real content — "thulhadhoo dhaairage MP Abdul Hannan sponsor koh" lost the
member's name entirely, silently, in the middle of a paragraph.

The chunker lives at module scope in app.py (not nested in the SSE generator)
specifically so it can be tested here. app.py eager-loads both models at import
so gunicorn workers are warm before serving; SKIP_MODEL_PRELOAD (set below,
before the import) suppresses that, so these tests need no weights at all.

Run: python tests/test_chunking.py
"""

import os
import sys
from pathlib import Path

os.environ['SKIP_MODEL_PRELOAD'] = '1'  # must precede the app import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from app import (
    CHUNK_WORDS,
    OVERLAP_WORDS,
    drop_repeated_prefix,
    split_into_word_chunks,
)


def words(n):
    return [f"w{i}" for i in range(n)]


def test_chunks_cover_every_word_exactly_once():
    """With no overlap the chunks must tile the input: no gaps, no repeats."""
    for n in (11, 25, 40, 61, 200):
        ws = words(n)
        chunks = split_into_word_chunks(" ".join(ws))
        rebuilt = " ".join(c for c, _ in chunks).split()
        assert rebuilt == ws, f"n={n}: {len(rebuilt)} words out of {n} in"


def test_first_chunk_is_flagged_and_only_the_first():
    chunks = split_into_word_chunks(" ".join(words(35)))
    flags = [is_first for _, is_first in chunks]
    assert flags[0] is True
    assert not any(flags[1:]), "only the first chunk may be flagged is_first"


def test_no_chunk_is_contained_in_the_previous_one():
    """The stride guard.

    Without the early break, 35 words at stride 16 emitted a final 3-word chunk
    covering words 32-34 — already wholly inside the 16-34 chunk. It was then too
    short to trim, so it got duplicated into the output.
    """
    for n in range(CHUNK_WORDS + 1, 4 * CHUNK_WORDS):
        chunks = split_into_word_chunks(" ".join(words(n)))
        seen = set()
        for text, _ in chunks:
            ws = set(text.split())
            assert not (ws & seen), f"n={n}: chunk repeats {sorted(ws & seen)}"
            seen |= ws


def test_no_chunk_exceeds_the_window():
    """Oversized chunks are what made the model start dropping content."""
    for n in (11, 25, 40, 61, 200):
        for text, _ in split_into_word_chunks(" ".join(words(n))):
            assert len(text.split()) <= CHUNK_WORDS, f"n={n}: {text}"


def test_identity_decode_reconstructs_the_input():
    """Stitching mirrors app.py: concatenate, de-duplicating against the previous
    chunk. With an identity 'decode' that must reproduce the input exactly — the
    property the old fixed-width trim violated."""
    for n in (11, 25, 37, 64, 200):
        ws = words(n)
        chunks = split_into_word_chunks(" ".join(ws))
        out = []
        for text, is_first in chunks:
            if not is_first and out:
                text = drop_repeated_prefix(out[-1], text)
            out.append(text)
        assert " ".join(out).split() == ws, f"failed at {n} words"


def test_short_phrases_are_not_chunked():
    text = " ".join(words(CHUNK_WORDS))
    chunks = split_into_word_chunks(text)
    assert chunks == [(text, True)]


def test_drop_repeated_prefix_is_a_noop_at_zero_overlap():
    assert OVERLAP_WORDS == 0, "these tests assume the shipped config"
    assert drop_repeated_prefix("a b c", "c d e") == "c d e"


def test_drop_repeated_prefix_trims_only_a_real_repeat():
    """Guards the config if OVERLAP_WORDS is ever raised again: trim what
    actually repeats, never a fixed count."""
    assert drop_repeated_prefix("a b c d", "c d e f", max_overlap=4) == "e f"
    # Nothing repeats -> drop nothing. A visible duplicate beats a silent delete.
    assert drop_repeated_prefix("a b c d", "x y z", max_overlap=4) == "x y z"
    # Compressed overlap: only 1 of the 2 overlapped words survived the model.
    assert drop_repeated_prefix("a b c d", "d e f", max_overlap=4) == "e f"


def test_empty_input():
    assert split_into_word_chunks("") == []
    assert split_into_word_chunks("   ") == []


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    print("\n" + ("all passed" if not failures else f"{failures} failed"))
    sys.exit(1 if failures else 0)
