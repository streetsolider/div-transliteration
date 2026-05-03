"""Thaana <-> Segha-phonetic keymap converter.

The Segha layout is the Standard Phonetic Dhivehi keyboard layout, taken
verbatim from jawish/jtk's `_transToKbd['phonetic']`. Every Thaana codepoint
used in everyday Dhivehi maps to a single ASCII byte, so the keymap form
costs ~3x fewer bytes than UTF-8 Thaana — the win we want for ByT5 output.

Round-trippable for all in-layout characters. Characters outside the
layout (digits, ASCII punctuation, whitespace) pass through unchanged.
"""

# Verbatim from jawish/jtk, in QWERTY scan order: row1 (q-p + []\), row2
# (a-l + ;'), row3 (z-m + ,./). The Thaana strings index the same positions.
_QWERTY_LOWER = "qwertyuiop[]\\asdfghjkl;'zxcvbnm,./"
_SEGHA_LOWER  = "ްއެރތޔުިޮޕ][\\ަސދފގހޖކލ؛'ޒ×ޗވބނމ،./"

_QWERTY_UPPER = "QWERTYUIOP{}|ASDFGHJKL:\"ZXCVBNM<>?)("
_SEGHA_UPPER  = "ޤޢޭޜޓޠޫީޯ÷}{|ާށޑﷲޣޙޛޚޅ:\"ޡޘޝޥޞޏޟ><؟)("

# Cross-script characters outside the Thaana block that are still real
# Thaana-side mappings (Arabic punctuation, Allah ligature, math symbols).
# Anything not matching this OR the Thaana block is treated as a layout
# passthrough/quirk (e.g. the [<->] bracket flip) and skipped — applying
# those flips at the text level would corrupt literal punctuation.
_CROSS_SCRIPT_MAPPINGS = {'،', '؛', '؟', 'ﷲ', '×', '÷'}


def _is_thaana(c: str) -> bool:
    return 'ހ' <= c <= '޿'


def _build_tables():
    thaana_to_latin = {}
    latin_to_thaana = {}
    for latin_row, thaana_row in [
        (_QWERTY_LOWER, _SEGHA_LOWER),
        (_QWERTY_UPPER, _SEGHA_UPPER),
    ]:
        assert len(latin_row) == len(thaana_row), \
            f"Layout length mismatch: {len(latin_row)} vs {len(thaana_row)}"
        for latin, thaana in zip(latin_row, thaana_row):
            if not (_is_thaana(thaana) or thaana in _CROSS_SCRIPT_MAPPINGS):
                continue
            thaana_to_latin[thaana] = latin
            latin_to_thaana[latin] = thaana
    return thaana_to_latin, latin_to_thaana


_THAANA_TO_LATIN, _LATIN_TO_THAANA = _build_tables()


def thaana_to_keymap(thaana: str) -> str:
    """Convert Thaana text to Segha-phonetic ASCII keymap."""
    return "".join(_THAANA_TO_LATIN.get(c, c) for c in thaana)


def keymap_to_thaana(keymap: str) -> str:
    """Convert Segha-phonetic ASCII keymap text back to Thaana."""
    return "".join(_LATIN_TO_THAANA.get(c, c) for c in keymap)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    cases = [
        ("އަންގާރަ",  "wanqgAra"),    # Tuesday
        ("ބަސް",      "basq"),         # language / bus
        ("ކުރު",      "kuru"),         # short
        ("ކަރު",      "karu"),         # throat
        ("ބޮއްކުރާ",  "bowqkurA"),     # gourd-like fruit
        ("ބަސް،",     "basq,"),        # Arabic comma -> ASCII
        ("ބަސް 123",  "basq 123"),     # digits/space pass through
        ("(ކުރު)",    "(kuru)"),       # parens pass through unflipped
    ]
    failed = 0
    for thaana, expected in cases:
        actual = thaana_to_keymap(thaana)
        roundtrip = keymap_to_thaana(actual)
        ok_fwd = actual == expected
        ok_rt = roundtrip == thaana
        flag = "OK" if (ok_fwd and ok_rt) else "FAIL"
        if not (ok_fwd and ok_rt):
            failed += 1
        print(f"[{flag}] {thaana!r:30} -> {actual!r:25} -> {roundtrip!r}")
        if not ok_fwd:
            print(f"       expected forward: {expected!r}")
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    print(f"Table sizes: thaana->latin={len(_THAANA_TO_LATIN)}, latin->thaana={len(_LATIN_TO_THAANA)}")
