"""Round-trip validation for keymap.py against the full source corpus.

For every (latin, dh) row in alakxender/dhivehi-transliteration-pairs:
  assert keymap_to_thaana(thaana_to_keymap(normalize(dh))) == normalize(dh)

Normalization mirrors what app.py already does at output time: rewrite
stray ASCII punctuation in Thaana text to its Arabic equivalents
(`,`->`،`, `;`->`؛`, `?`->`؟`). The source corpus has these mixed in.

Reports raw + normalized pass rates and the top offending characters.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from datasets import load_dataset

from keymap import keymap_to_thaana, thaana_to_keymap


def normalize(thaana: str) -> str:
    """Mirror app.py:238-240 — rewrite stray ASCII punctuation to Arabic."""
    return thaana.replace(",", "،").replace(";", "؛").replace("?", "؟")


def diff_chars(original: str, roundtripped: str) -> list[tuple[int, str, str]]:
    """Return (position, original_char, roundtripped_char) for each mismatch."""
    diffs = []
    for i, (a, b) in enumerate(zip(original, roundtripped)):
        if a != b:
            diffs.append((i, a, b))
    if len(original) != len(roundtripped):
        diffs.append((min(len(original), len(roundtripped)), "<len mismatch>", ""))
    return diffs


def run(split_name: str, ds) -> tuple[int, int, int, Counter, list]:
    total = len(ds)
    raw_failures = 0
    norm_failures = 0
    offenders = Counter()
    sample_failures = []
    for i, row in enumerate(ds):
        thaana = row["dh"]

        # Raw round-trip (source corpus as-is)
        if keymap_to_thaana(thaana_to_keymap(thaana)) != thaana:
            raw_failures += 1

        # Normalized round-trip (what we'll actually train on)
        norm = normalize(thaana)
        keymap = thaana_to_keymap(norm)
        roundtrip = keymap_to_thaana(keymap)
        if roundtrip != norm:
            norm_failures += 1
            for _pos, orig, _rt in diff_chars(norm, roundtrip):
                if orig != "<len mismatch>":
                    offenders[orig] += 1
            if len(sample_failures) < 5:
                sample_failures.append((norm, keymap, roundtrip))

        if (i + 1) % 10000 == 0:
            print(f"  [{split_name}] {i+1}/{total} processed, raw fails={raw_failures}, norm fails={norm_failures}")
    return total, raw_failures, norm_failures, offenders, sample_failures


def main():
    print("Loading dataset alakxender/dhivehi-transliteration-pairs ...")
    ds = load_dataset("alakxender/dhivehi-transliteration-pairs")
    grand_offenders = Counter()
    grand_total = 0
    grand_raw_failures = 0
    grand_norm_failures = 0

    for split in ["train", "test"]:
        print(f"\n=== {split} split ({len(ds[split])} rows) ===")
        total, raw_failures, norm_failures, offenders, samples = run(split, ds[split])
        grand_total += total
        grand_raw_failures += raw_failures
        grand_norm_failures += norm_failures
        grand_offenders.update(offenders)
        raw_rate = (raw_failures / total * 100) if total else 0.0
        norm_rate = (norm_failures / total * 100) if total else 0.0
        print(f"  Raw round-trip:        {raw_failures}/{total} failed ({raw_rate:.2f}%)")
        print(f"  Normalized round-trip: {norm_failures}/{total} failed ({norm_rate:.4f}%)")
        if samples:
            print(f"  First normalized-fail samples (showing up to 5):")
            for norm, keymap, roundtrip in samples:
                print(f"    norm:      {norm!r}")
                print(f"    keymap:    {keymap!r}")
                print(f"    roundtrip: {roundtrip!r}")
                print()

    print(f"\n=== Total ===")
    print(f"  Raw:        {grand_raw_failures}/{grand_total} failed ({grand_raw_failures/grand_total*100:.2f}%)")
    print(f"  Normalized: {grand_norm_failures}/{grand_total} failed ({grand_norm_failures/grand_total*100:.4f}%)")
    if grand_offenders:
        print(f"\n=== Top 30 offending characters in normalized round-trip ===")
        for char, count in grand_offenders.most_common(30):
            codepoint = f"U+{ord(char):04X}"
            print(f"  {char!r:>6}  {codepoint}  count={count}")

    if grand_norm_failures > 0:
        print(f"\nFAIL: normalized round-trip should be 100%")
        sys.exit(1)
    else:
        print(f"\nPASS: normalized round-trip is 100% on {grand_total} rows")


if __name__ == "__main__":
    main()
