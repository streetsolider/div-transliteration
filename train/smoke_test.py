"""Smoke test: load Neobe checkpoint on GPU and run one inference.

Verifies the CUDA + Blackwell stack works end-to-end with this model
before we commit to a multi-hour fine-tuning run. Mirrors app.py's
generate() params so timing is comparable to the production pipeline.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

CHECKPOINT = "Neobe/dhivehi-byt5-latin2thaana-v1"
SAMPLES = [
    "Aharennakee Dhivehi bahun vaahaka dhakkaa meeheh",
    "salaam dhivehi raajje",
    "kuru",  # MODEL_NOTES failing case
    "bohkuraa",  # MODEL_NOTES failing case
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nLoading {CHECKPOINT} ...")
    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT).to(device)
    model.eval()
    load_time = time.perf_counter() - t0
    print(f"Loaded in {load_time:.1f}s")
    if device.type == "cuda":
        print(f"VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated")

    print(f"\n=== Inference (matching app.py:202-211 params) ===")
    inputs = tokenizer(SAMPLES, return_tensors="pt", padding=True, truncation=False).to(device)

    # Warmup
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=64, num_beams=4, do_sample=False)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Timed run
    t0 = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=4,
            do_sample=False,
            early_stopping=False,
            length_penalty=1.2,
            repetition_penalty=1.0,
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    print(f"Batch of {len(SAMPLES)} inputs decoded in {elapsed*1000:.0f}ms ({elapsed*1000/len(SAMPLES):.0f}ms/sample)")
    if device.type == "cuda":
        print(f"VRAM peak during inference: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    print()
    for latin, thaana in zip(SAMPLES, decoded):
        print(f"  in:  {latin!r}")
        print(f"  out: {thaana!r}")
        print()


if __name__ == "__main__":
    main()
