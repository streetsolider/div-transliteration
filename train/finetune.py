"""Fine-tune Neobe ByT5 latin2thaana checkpoint to emit Segha keymap.

Warm-starts from Neobe/dhivehi-byt5-latin2thaana-v1 — encoder already
understands Latin Dhivehi orthography; only the decoder retargets from
Thaana bytes to ASCII Segha keymap. Trains on train/data/keymap_pairs/
(see train/build_corpus.py).

Usage:
  python train/finetune.py [--resume]

Output: best checkpoint saved to train/checkpoints/byt5-latin2keymap/.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

BASE_CHECKPOINT = "Neobe/dhivehi-byt5-latin2thaana-v1"
DATA_DIR = Path(__file__).resolve().parent / "data" / "keymap_pairs"
OUTPUT_DIR = Path(__file__).resolve().parent / "checkpoints" / "byt5-latin2keymap"

# Corpus stats: p95 latin/keymap = 97/99 bytes, p99 = 946/977. Capping at 512
# truncates only the rare long-concatenation augmentations, but those still
# contribute partial long-sequence signal up to the cap.
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512
EVAL_SUBSET_SIZE = 1000  # Sample of test split used during training (full eval is separate)
# Filter out long-concat augmentations for the first fine-tune. Combined
# input+target length cap; base corpus is ~360 bytes max combined, single-word
# augmentations are tiny, so 400 keeps 95%+ of data. Re-enable long-concats
# for a follow-up training round if eval shows long-input degradation.
MAX_TRAIN_COMBINED_LENGTH = 400


def tokenize(batch, tokenizer):
    inputs = tokenizer(
        batch["latin"], max_length=MAX_INPUT_LENGTH, truncation=True,
    )
    labels = tokenizer(
        text_target=batch["keymap"], max_length=MAX_TARGET_LENGTH, truncation=True,
    )
    inputs["labels"] = labels["input_ids"]
    # Length column for group_by_length — use combined input+target since
    # decoder activations scale with target length too.
    inputs["length"] = [
        len(i) + len(l) for i, l in zip(inputs["input_ids"], labels["input_ids"])
    ]
    return inputs


def make_compute_metrics(tokenizer):
    pad_id = tokenizer.pad_token_id

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        # Trainer fills label padding with -100; restore for decoding.
        labels = np.where(labels != -100, labels, pad_id)
        # preds may be -100 too if generation_max_length was hit before EOS
        preds = np.where(preds != -100, preds, pad_id)
        pred_str = tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

        exact = sum(p == l for p, l in zip(pred_str, label_str)) / max(len(pred_str), 1)

        # Character-level accuracy via simple normalized edit distance
        def char_acc(pred, label):
            if not label and not pred:
                return 1.0
            n, m = len(pred), len(label)
            if n == 0 or m == 0:
                return 0.0
            # Edit-distance DP — bounded by min(n, m) so OK for our sizes
            prev = list(range(m + 1))
            for i in range(1, n + 1):
                cur = [i] + [0] * m
                for j in range(1, m + 1):
                    cost = 0 if pred[i - 1] == label[j - 1] else 1
                    cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
                prev = cur
            return 1.0 - prev[m] / max(n, m)

        char_accs = [char_acc(p, l) for p, l in zip(pred_str, label_str)]
        return {
            "exact_match": exact,
            "char_acc": sum(char_accs) / max(len(char_accs), 1),
        }

    return compute_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--smoke", action="store_true",
                        help="Run 50 training steps + 1 eval, then exit. For pipeline validation.")
    args = parser.parse_args()

    print(f"CUDA: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    print(f"\nLoading tokenizer + model from {BASE_CHECKPOINT} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_CHECKPOINT)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_CHECKPOINT)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    print(f"\nLoading dataset from {DATA_DIR} ...")
    ds = load_from_disk(str(DATA_DIR))
    print(f"  train: {len(ds['train'])} pairs")
    print(f"  test:  {len(ds['test'])} pairs")

    # Subsample eval to keep during-training evaluation fast.
    eval_size = 200 if args.smoke else EVAL_SUBSET_SIZE
    eval_ds = ds["test"].shuffle(seed=42).select(range(eval_size))
    train_ds = ds["train"]
    if args.smoke:
        train_ds = train_ds.shuffle(seed=42).select(range(2000))
        print(f"  SMOKE MODE: train={len(train_ds)}, eval={len(eval_ds)}")
    else:
        print(f"  eval subset: {len(eval_ds)} pairs (full eval runs separately)")

    print(f"\nTokenizing ...")
    train_tok = train_ds.map(
        lambda b: tokenize(b, tokenizer),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    eval_tok = eval_ds.map(
        lambda b: tokenize(b, tokenizer),
        batched=True,
        remove_columns=eval_ds.column_names,
    )

    pre_filter = len(train_tok)
    train_tok = train_tok.filter(lambda x: x["length"] <= MAX_TRAIN_COMBINED_LENGTH)
    print(f"  filtered to length<={MAX_TRAIN_COMBINED_LENGTH}: {len(train_tok)}/{pre_filter} kept")

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        padding="longest",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = OUTPUT_DIR / "smoke" if args.smoke else OUTPUT_DIR
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        overwrite_output_dir=args.smoke,
        num_train_epochs=3 if not args.smoke else 1,
        max_steps=50 if args.smoke else -1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,  # effective batch = 32
        learning_rate=3e-5,
        warmup_steps=10 if args.smoke else 500,
        weight_decay=0.01,
        bf16=True,
        group_by_length=True,  # huge speedup — keeps each batch's max length tight
        length_column_name="length",
        eval_strategy="steps",
        eval_steps=50 if args.smoke else 500,
        save_strategy="no" if args.smoke else "steps",
        save_steps=1000,
        save_total_limit=3,
        load_best_model_at_end=not args.smoke,
        metric_for_best_model="char_acc",
        greater_is_better=True,
        logging_steps=10 if args.smoke else 50,
        predict_with_generate=True,
        generation_max_length=256 if args.smoke else MAX_TARGET_LENGTH,
        generation_num_beams=1,  # greedy during eval; beams slow without quality win at this scale
        report_to="none",
        dataloader_num_workers=0,  # Windows-safe
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=make_compute_metrics(tokenizer),
    )

    print(f"\nStarting fine-tune ...")
    print(f"  steps/epoch: {len(train_tok) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")
    print(f"  total steps: ~{len(train_tok) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")

    trainer.train(resume_from_checkpoint=args.resume)

    if args.smoke:
        print(f"\nSmoke run complete — pipeline validated. Skipping save.")
        return

    print(f"\nSaving best model to {OUTPUT_DIR} ...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Done.")


if __name__ == "__main__":
    main()
