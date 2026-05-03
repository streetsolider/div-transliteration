"""Build a length-diverse (latin, keymap) training corpus for fine-tuning.

Sources from alakxender/dhivehi-transliteration-pairs and:
1. Normalizes ASCII punctuation in the Thaana column to Arabic equivalents
   (mirrors app.py:238-240) — required for round-trip integrity (see
   tests/test_keymap_roundtrip.py).
2. Converts the Thaana column to Segha-keymap via thaana_to_keymap.
3. Augments train split with:
   - Word-aligned single-word pairs from word-count-parity rows in the
     corpus (fixes the headline-only short-input hallucination per
     MODEL_NOTES.md).
   - Long concatenated pairs (paragraph-length, capped to fit under
     ByT5-small's 1024 max_position_embeddings).

Output: HuggingFace DatasetDict saved to train/data/keymap_pairs/ with
splits 'train' and 'test', schema {latin: str, keymap: str}.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from datasets import Dataset, DatasetDict, load_dataset

from keymap import keymap_to_thaana, thaana_to_keymap

# ByT5-small max_position_embeddings = 1024. Cap both sides under that with
# headroom for special tokens. Keymap is ~1.05-1.2x Latin (Latin elides some
# vowels that Thaana writes explicitly), so we must check both, not just input.
MAX_SEQ_BYTES = 1000
SHORT_AUG_FRACTION = 0.03
LONG_AUG_FRACTION = 0.03
RANDOM_SEED = 42


def normalize(thaana: str) -> str:
    """ASCII -> Arabic punctuation in Thaana text. Mirrors app.py:238-240."""
    return thaana.replace(",", "،").replace(";", "؛").replace("?", "؟")


_LATIN_STRIP = ".,!?:;\"'()[]{}"
_THAANA_STRIP = "،؛؟.!:\"'()[]{}"


def extract_single_word_pairs(rows) -> list[tuple[str, str]]:
    """Single-word pairs from word-count-parity rows. Conservative: drops
    rows where Latin and Thaana side have different whitespace-token counts,
    since misaligned tokens would produce garbage pairs."""
    pairs = set()
    parity = 0
    total = 0
    for row in rows:
        total += 1
        latin_words = row["latin"].split()
        dh_words = normalize(row["dh"]).split()
        if len(latin_words) != len(dh_words):
            continue
        parity += 1
        for lw, dw in zip(latin_words, dh_words):
            lw = lw.strip(_LATIN_STRIP)
            dw = dw.strip(_THAANA_STRIP)
            if lw and dw:
                pairs.add((lw, dw))
    print(f"    word-count parity: {parity}/{total} rows ({parity/total*100:.1f}%)")
    return list(pairs)


def build_long_concatenations(rows, target_count: int, rng) -> list[tuple[str, str]]:
    """Concatenate consecutive rows until either Latin or keymap-converted
    bytes approach MAX_SEQ_BYTES. Both sides must fit under the model cap."""
    rows = list(rows)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    pairs = []
    i = 0
    while len(pairs) < target_count and i < len(indices):
        latin_chunks = []
        dh_chunks = []
        latin_bytes = 0
        keymap_bytes = 0
        while i < len(indices):
            row = rows[indices[i]]
            latin = row["latin"]
            dh = normalize(row["dh"])
            keymap = thaana_to_keymap(dh)
            sep = 1 if latin_chunks else 0
            new_latin = latin_bytes + sep + len(latin.encode("utf-8"))
            new_keymap = keymap_bytes + sep + len(keymap.encode("utf-8"))
            if new_latin > MAX_SEQ_BYTES or new_keymap > MAX_SEQ_BYTES:
                break
            latin_chunks.append(latin)
            dh_chunks.append(dh)
            latin_bytes = new_latin
            keymap_bytes = new_keymap
            i += 1
        if len(latin_chunks) >= 2:
            pairs.append((" ".join(latin_chunks), " ".join(dh_chunks)))
    return pairs


def to_dataset(pairs: list[tuple[str, str]]) -> Dataset:
    return Dataset.from_dict({
        "latin":  [p[0] for p in pairs],
        "keymap": [p[1] for p in pairs],
    })


def percentiles(values: list[int]) -> str:
    n = len(values)
    s = sorted(values)
    p = lambda f: s[min(int(n * f), n - 1)]
    return (f"min={s[0]}  p25={p(0.25)}  p50={p(0.5)}  p75={p(0.75)}  "
            f"p95={p(0.95)}  p99={p(0.99)}  max={s[-1]}")


def main():
    rng = random.Random(RANDOM_SEED)
    print("Loading source corpus ...")
    src = load_dataset("alakxender/dhivehi-transliteration-pairs")

    splits = {}
    for split_name in ["train", "test"]:
        split = src[split_name]
        print(f"\n=== {split_name} split ({len(split)} source rows) ===")

        base = []
        for row in split:
            latin = row["latin"]
            keymap = thaana_to_keymap(normalize(row["dh"]))
            base.append((latin, keymap))
        print(f"  base pairs: {len(base)}")

        pairs = list(base)

        if split_name == "train":
            # Word-aligned single-word pairs — fills the short-input gap
            single_word_thaana = extract_single_word_pairs(split)
            single_word = [(lw, thaana_to_keymap(dw)) for lw, dw in single_word_thaana]
            target_short = max(1, int(len(base) * SHORT_AUG_FRACTION))
            if len(single_word) > target_short:
                rng.shuffle(single_word)
                single_word = single_word[:target_short]
            pairs.extend(single_word)
            print(f"  + single-word pairs: {len(single_word)}  (target {target_short}, available {len(single_word_thaana)})")

            # Long concatenations — fills the long-sequence gap
            target_long = max(1, int(len(base) * LONG_AUG_FRACTION))
            long_thaana = build_long_concatenations(split, target_long, rng)
            long_pairs = [(lat, thaana_to_keymap(dh)) for lat, dh in long_thaana]
            pairs.extend(long_pairs)
            print(f"  + long concatenated pairs: {len(long_pairs)}  (target {target_long})")

        splits[split_name] = to_dataset(pairs)
        print(f"  total: {len(splits[split_name])} pairs")

    ds = DatasetDict(splits)
    out_dir = Path(__file__).resolve().parent / "data" / "keymap_pairs"
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    print(f"\nSaved to {out_dir}")
    print(f"Splits: {dict((k, len(v)) for k, v in ds.items())}")

    print(f"\n=== train length distribution (bytes) ===")
    latin_bytes = [len(s.encode("utf-8")) for s in ds["train"]["latin"]]
    keymap_bytes = [len(s.encode("utf-8")) for s in ds["train"]["keymap"]]
    print(f"  latin:  {percentiles(latin_bytes)}")
    print(f"  keymap: {percentiles(keymap_bytes)}")
    over_latin = sum(1 for b in latin_bytes if b > MAX_SEQ_BYTES)
    over_keymap = sum(1 for b in keymap_bytes if b > MAX_SEQ_BYTES)
    if over_latin or over_keymap:
        print(f"  WARNING: {over_latin} latin / {over_keymap} keymap rows exceed MAX_SEQ_BYTES={MAX_SEQ_BYTES}")

    print(f"\n=== round-trip spot check (50 random train rows) ===")
    sample_indices = rng.sample(range(len(ds["train"])), 50)
    fails = 0
    for i in sample_indices:
        keymap = ds["train"][i]["keymap"]
        thaana = keymap_to_thaana(keymap)
        if thaana_to_keymap(thaana) != keymap:
            fails += 1
    print(f"  {50 - fails}/50 keymap round-trip OK")


if __name__ == "__main__":
    main()
