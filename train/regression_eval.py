"""Side-by-side regression eval: new keymap-output model vs. shipped Neobe.

Runs the MODEL_NOTES.md canonical failing cases plus realistic sentences
through both models with production decoder params, decodes the new
model's keymap output back to Thaana, and prints a comparison.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from keymap import keymap_to_thaana

NEW_CHECKPOINT = "str33t/dhivehi-byt5-latin2thaana-keymap-v1"
BASELINE_CHECKPOINT = "Neobe/dhivehi-byt5-latin2thaana-v1"

# (latin_input, expected_thaana_or_None)
CASES = [
    # MODEL_NOTES.md failing cases
    ("bohkuraa", "ބޮއްކުރާ"),
    ("kuru",     "ކުރު"),
    ("karu",     "ކަރު"),
    ("bas",      "ބަސް"),
    # Sanity-check realistic sentences (no expected; eyeball the output)
    ("Aharennakee Dhivehi bahun vaahaka dhakkaa meeheh", None),
    ("Salaam dhivehi raajje", None),
    ("Maadhama haveeruge bahdhaluvun cancel kohffi", None),
]

GEN_KWARGS = dict(
    max_new_tokens=256,
    num_beams=4,
    do_sample=False,
    early_stopping=False,
    length_penalty=1.0,  # reverted from 1.2; new model shouldn't need the hack
    repetition_penalty=1.0,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(ckpt: str):
    tok = AutoTokenizer.from_pretrained(ckpt)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(DEVICE).eval()
    return tok, mdl


def infer(tok, mdl, texts, decode_keymap: bool):
    inputs = tok(texts, return_tensors="pt", padding=True, truncation=False).to(DEVICE)
    with torch.inference_mode():
        out = mdl.generate(**inputs, **GEN_KWARGS)
    decoded = tok.batch_decode(out, skip_special_tokens=True)
    if decode_keymap:
        decoded = [keymap_to_thaana(s) for s in decoded]
    return decoded


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading baseline: {BASELINE_CHECKPOINT}")
    base_tok, base_mdl = load(BASELINE_CHECKPOINT)
    print(f"Loading new:      {NEW_CHECKPOINT}")
    new_tok, new_mdl = load(NEW_CHECKPOINT)

    inputs = [c[0] for c in CASES]

    t0 = time.perf_counter()
    base_out = infer(base_tok, base_mdl, inputs, decode_keymap=False)
    base_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    new_out_keymap = infer(new_tok, new_mdl, inputs, decode_keymap=False)
    new_out_thaana = [keymap_to_thaana(s) for s in new_out_keymap]
    new_time = time.perf_counter() - t0

    print(f"\n=== Inference time on {len(inputs)} inputs ===")
    print(f"  baseline:  {base_time*1000:.0f}ms ({base_time*1000/len(inputs):.0f}ms/input)")
    print(f"  new:       {new_time*1000:.0f}ms ({new_time*1000/len(inputs):.0f}ms/input)")
    print(f"  speedup:   {base_time/new_time:.2f}x")

    print(f"\n=== Output byte counts (avg) ===")
    base_bytes = sum(len(s.encode("utf-8")) for s in base_out) / len(base_out)
    new_keymap_bytes = sum(len(s.encode("utf-8")) for s in new_out_keymap) / len(new_out_keymap)
    print(f"  baseline (Thaana UTF-8): {base_bytes:.1f} bytes/output")
    print(f"  new (ASCII keymap):      {new_keymap_bytes:.1f} bytes/output")
    print(f"  ratio:                   {base_bytes/new_keymap_bytes:.2f}x")

    print(f"\n=== Per-case comparison ===")
    pass_count = 0
    fail_count = 0
    for (latin, expected), b, kn, nt in zip(CASES, base_out, new_out_keymap, new_out_thaana):
        print(f"\nINPUT: {latin!r}")
        print(f"  baseline:    {b!r}")
        print(f"  new keymap:  {kn!r}")
        print(f"  new -> dh:   {nt!r}")
        if expected is not None:
            print(f"  expected:    {expected!r}")
            base_ok = b == expected
            new_ok = nt == expected
            print(f"  baseline {'PASS' if base_ok else 'FAIL'}   |   new {'PASS' if new_ok else 'FAIL'}")
            if new_ok:
                pass_count += 1
            else:
                fail_count += 1

    print(f"\n=== Summary on cases with expected output ===")
    print(f"  new model: {pass_count}/{pass_count+fail_count} pass")


if __name__ == "__main__":
    main()
