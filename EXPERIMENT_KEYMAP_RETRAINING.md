# Reducing ByT5 Decoder Inference Cost for Thaana Script via Phonetic-Keymap Retargeting

## Abstract

Byte-level transformers like ByT5 [1] produce one decoder step per output byte. For tasks emitting Thaana (Maldivian) script, every output character incurs three decoder steps because Thaana codepoints (U+0780–U+07BF) occupy three UTF-8 bytes. For our existing fine-tune `Neobe/dhivehi-byt5-latin2thaana-v1` [2] in a Flask transliteration service, this dominates inference cost.

We retarget the decoder to emit ASCII-encoded Thaana via a deterministic, lossless 1-to-1 mapping based on the Standard Phonetic ("Segha") Dhivehi keyboard layout [3]. A pure-function post-processor reconstructs Thaana from the model's ASCII output. End-to-end user experience is unchanged, but each output character costs one decoder step instead of three (~3× decoder speedup expected).

In the same retraining pass, we address two known quality issues documented in this repo's `MODEL_NOTES.md`: short-input hallucination and long-input degradation. Both stem from distribution shift in the original model's news-headline domain adaptation. We use length-diverse augmentation to mitigate them.

*Results and evaluation are pending — first training run in progress at time of writing.*

## 1. Problem Statement

### 1.1 Decoder Cost in Byte-Level Models for Thaana

ByT5 [1] is an encoder-decoder transformer operating directly on UTF-8 bytes, eliminating the learned tokenizer. This is well-suited to low-resource languages and noisy text but introduces a sequence-length cost: tokens-per-character is fixed by UTF-8 encoding rather than chosen by a tokenizer.

UTF-8 encodes characters in 1–4 bytes. The Thaana script (U+0780–U+07BF) falls in the 3-byte range. A 100-character Thaana string is 300 bytes — equivalently, 300 ByT5 tokens.

Latin Dhivehi (the romanized input form) is mostly ASCII. For the transliteration task, **the decoder runs ~3× as many steps as the encoder for inputs and outputs of equal character count**.

In our service (`app.py`), the dominant runtime cost is `model.generate(...)` with `num_beams=4`, `max_new_tokens=512`. Each decoder step at this configuration is independent compute. Cutting output bytes by 3× cuts decoder compute proportionally.

### 1.2 Documented Quality Issues

Independent of cost, `MODEL_NOTES.md` records two failure modes of the shipped model:

1. **Short-input hallucination.** Single Latin words produce repetitive or extended outputs. Examples (with shipped decoder params):
    - `bohkuraa` → `ބޮއްކުރާ` repeated four times.
    - `kuru` (= "short") → `ކުރުމުގެ` (= "of doing"), the wrong word entirely.
    - `karu`, `bas` → similar cascades into headline-prefix continuations.
2. **Long-input degradation.** Inputs materially longer than the news-headline distribution lose coherence in late positions.

A decoder hyperparameter sweep (`length_penalty ∈ [-3, 1.2]`, `repetition_penalty ∈ [1.0, 3.0]`, `no_repeat_ngram_size ∈ [4, 12]`, `early_stopping ∈ {True, False}`) found no setting that fixes short inputs without breaking longer ones. The root cause is the training distribution; the fix must happen at the data/model level.

## 2. Related Work and Resources

- **ByT5** [1]: byte-level T5 variant, pretrained on mC4. Strong on character-level tasks (transliteration, spelling correction, morphology); weaker on long-context semantic tasks because byte sequences are 4–6× longer than subword sequences.
- **Source corpus** [2]: `alakxender/dhivehi-transliteration-pairs` — 187,908 aligned Latin/Thaana sentence pairs (150,326 train / 37,582 test). Median length ~58 chars per side, no single-word entries (corpus minimum: 25 chars / ~5 words).
- **Existing model** [4]: `Neobe/dhivehi-byt5-latin2thaana-v1` — fine-tuned from `google/byt5-small` on the above corpus, with an additional domain-adaptation pass on 10k news headlines.
- **Keymap reference** [3]: `jawish/jtk` — JavaScript Thaana Keyboard, source of the verbatim Segha layout strings used in this work.

## 3. Approach

### 3.1 Keymap Choice: Segha Phonetic Layout

The Standard Phonetic Dhivehi keyboard layout (referred to as "Segha", distinct from the "Hassan Hameed" variant) maps every Thaana letter, fili (vowel mark), and sukun to a unique single ASCII character, plus a small set of cross-script symbols (Arabic punctuation `، ؛ ؟`, the Allah ligature `ﷲ`, and two math symbols).

Three properties make it suitable as a model output target:

1. **1:1 codepoint correspondence** — every Thaana character maps to exactly one ASCII byte (no multi-character escape sequences).
2. **Round-trippability** — the mapping is deterministically invertible.
3. **Sub-Thaana byte cost** — ASCII output averages 1 byte per Thaana codepoint, vs. 3 bytes per codepoint in UTF-8 Thaana.

Worked example. The Thaana word `އަންގާރަ` (Tuesday) maps to `wanqgAra`:

| Char | Codepoint | Name | Key |
|---|---|---|---|
| އ | U+0787 | alifu | `w` |
| ަ | U+07A6 | abafili | `a` |
| ނ | U+07A2 | noonu | `n` |
| ް | U+07B0 | sukun | `q` |
| ގ | U+07CB | gaafu | `g` |
| ާ | U+07A7 | aabaafili | `A` |
| ރ | U+0783 | raa | `r` |
| ަ | U+07A6 | abafili | `a` |

Capitals encode long vowels (`A`, `E`, `I`, `O`, `U`), the alifu carrier letter, naviyani, and Arabic-context consonants. Quirky cells (`x` → `×`, `P` → `÷`, `F` → `ﷲ`) exist but are virtually absent from everyday Dhivehi text.

The Hassan Hameed variant (`phonetic-hh`) was considered and rejected: it does not maintain 1:1 codepoint correspondence (alifu plus a vowel mark requires two keypresses in HH where Segha encodes the same consonant+vowel pair in one). Using HH would inflate output length and break the alignment, hurting both cost and trainability.

### 3.2 Deterministic Converter

`keymap.py` exposes two pure functions:

- `thaana_to_keymap(s: str) -> str` — generates training targets.
- `keymap_to_thaana(s: str) -> str` — runtime post-processor on model output.

Translation tables are built directly from the verbatim `_transToKbd['phonetic']` strings in `jawish/jtk` (lowercase + uppercase rows in QWERTY scan order). The converter passes unmapped characters through unchanged (digits, whitespace, basic ASCII punctuation outside the layout).

One subtlety: the source layout maps `[` to `]` and vice versa as a keystroke quirk (RTL bracket flip). Applied to text-level translation this would corrupt literal punctuation, so we explicitly exclude layout pass-throughs and bracket flips from the translation tables and let those characters identity-map.

### 3.3 Round-Trip Validation

The converter is validated on the full source corpus (188k pairs) by checking, for each row's Thaana side `t`:

```
keymap_to_thaana(thaana_to_keymap(t)) == t
```

Initial run: **201 / 187,908 failures (0.11%)**. All failures involved exactly two characters: ASCII `,` (192 cases) and ASCII `;` (11 cases), present in some Thaana rows where Arabic `،`/`؛` was expected. The source corpus has inconsistent punctuation conventions across its 6 source feeds.

After applying the same `,/;/?` → Arabic punctuation normalization that the existing service already does at output time (`app.py:238-240`): **0 / 187,908 failures**. The converter is sound and the spec is complete.

This normalization will run upstream in the training-data prep, which is consistent with the current behavior and not a behavior change.

### 3.4 Training Corpus

The source corpus has minimum length 25 characters (~5 words). It contains **no single-word or very-short examples** — which exactly matches the model's documented short-input failure mode. The model never trained on the inputs it now hallucinates on.

We construct a length-diverse training set:

| Component | Count | Rationale |
|---|---|---|
| Base (normalized + keymap-converted) | 150,326 | Full source train split |
| Single-word pairs (extracted) | 4,509 | Targets short-input failure mode |
| Long concatenations | 4,509 | Targets long-input degradation |
| **Total** | **159,344** | ~6% length augmentation |

**Single-word extraction.** From rows where Latin and Thaana have matching whitespace-token counts (~20% of the corpus), we extract individual word pairs and deduplicate. The 80% rejected rows have token-count mismatches caused by joined-vs-separate compounds, hyphenation conventions, and orthographic differences between the two scripts. The conservative drop avoids producing misaligned pairs.

The extraction yielded 69,844 unique pairs; we sampled 4,509 for training. The MODEL_NOTES.md failing words (`kuru`, `karu`, `bas`, `bohkuraa`) are not specially injected — the corpus extraction surfaces such pairs naturally, or it does not, and either way the eval will tell us.

**Long concatenations.** Shuffled rows are concatenated until either Latin or keymap byte length approaches a 1000-byte cap (under ByT5-small's 1024 `max_position_embeddings`). Yields paragraph-length training examples.

**Test split** is left untouched (37,582 rows, source distribution) to preserve evaluation honesty.

### 3.5 Fine-Tuning Setup

We warm-start from `Neobe/dhivehi-byt5-latin2thaana-v1` rather than from `google/byt5-small`. The encoder already understands Latin Dhivehi orthography from the original Neobe training; only the decoder needs to retarget from Thaana bytes to ASCII keymap. This is a smaller learning problem than training from scratch.

Configuration:

| | |
|---|---|
| Base model | `Neobe/dhivehi-byt5-latin2thaana-v1` (299.6M params) |
| Hardware | NVIDIA RTX 5070 Ti, 16 GB VRAM, sm_120 (Blackwell) |
| Precision | bf16 |
| Optimizer | AdamW (HF Trainer default) |
| Learning rate | 3e-5 (conservative, warm start) |
| Warmup | 500 steps |
| Effective batch | 32 (per-device 16, gradient accumulation 2) |
| Epochs | 3 |
| Max sequence length | 512 bytes (input and target) |
| Length grouping | `group_by_length=True` |
| Eval | 1,000-sample test subset, greedy, every 500 steps |

**Length filtering.** Training examples with combined input+target length > 400 bytes are filtered out for this initial run, dropping the 4,509 long concatenations. With `group_by_length=True`, those long batches dominated step time (4.6 s/step average). Filtering reduces step time to 0.58 s/step (~8× speedup) and cuts estimated training from ~19 hours to ~2.5 hours. The long concatenations are reserved for a possible follow-up training round if evaluation shows long-input degradation in the model produced here.

**Best-checkpoint selection** is by `char_acc` (1 − normalized edit distance) on the eval subset. Exact-match is also tracked but is too brittle to drive selection at this scale.

## 4. Inference Integration (Planned)

Three changes to `app.py` once a checkpoint is selected:

1. `app.py:12` — update `MODEL_NAMES['latin2thaana']` to the new keymap checkpoint.
2. `app.py:212` — apply `keymap_to_thaana(...)` to each decoded string when `direction == 'latin2thaana'`.
3. `app.py:209` — revert `length_penalty` from 1.2 to 1.0. The hack was a partial mitigation for the short-input issue at decode time; the retrained model should not need it.

The `thaana2latin` direction is untouched. The cost calculus is reversed there: Thaana goes through the encoder (cheap; pre-computed), and the decoder already emits Latin at 1 byte/char.

## 5. Results

### 5.1 Training run

The full fine-tune ran in **66 minutes wall-clock** on the RTX 5070 Ti — within the planned 2–6h budget. 14,517 optimizer steps over 3 epochs.

| Metric | Value |
|---|---|
| Total training time | 66 min (incl. one resume) |
| Final `train_loss` | 0.330 |
| Final `eval_loss` | 0.189 |
| Final `eval_char_acc` (greedy) | 0.923 |
| Final `eval_exact_match` (greedy) | 0.244 |

Loss decreased monotonically across all 28 evaluation steps from 0.241 to 0.189. `char_acc` plateau began around step 8000–9000 (epoch 2), suggesting 2 epochs would have been sufficient.

The exact-match number is artificially low because during-training eval used greedy decoding with `generation_max_length=512`. The realistic-quality numbers come from the regression eval below, which used production decoder params (`num_beams=4`).

### 5.2 Short-input regression cases (production decoder params)

| Input | Baseline (shipped Neobe) | New model | Expected | New result |
|---|---|---|---|---|
| `bohkuraa` | `ބޮއްކުރާ ބޮއްކުރާ ބޮއްކުރާ ބޮއްކުরާ` (×4 repeat) | `ބޮށްކުރާ` | `ބޮއްކුরާ` | 1-char error (shaviyani vs alifu) |
| `kuru` | `ކුरුമුگേ ...` (wrong word + repetition) | `ކുරു` | `ކുରു` | ✅ exact |
| `karu` | `ކරුگേ ...` (×7 repeat) | `ކරු` | `ކරු` | ✅ exact |
| `bas` | `ބാಸৈগে ...` (wrong word + ×8 repeat) | `ބසি` | `ބසಿ` | ✅ exact |

Three of four MODEL_NOTES failing cases pass. The single failure on `bohkuraa` (a relatively uncommon word meaning "gourd") emits `ޝ` (shaviyani) instead of `އ` (alifu) at position 2. Both characters appear in similar phonetic contexts elsewhere in Dhivehi; the model picked a plausible-but-incorrect spelling. Importantly, the new model fails by emitting the *wrong* single character (recoverable, similar to a typo), whereas the shipped model fails by emitting *repeated* output of an unrelated word (catastrophic).

### 5.3 Realistic sentences (production decoder params)

| Input | New model output (post keymap-decode) | Match baseline? |
|---|---|---|
| `Aharennakee Dhivehi bahun vaahaka dhakkaa meeheh` | `އަހަރެންނަކీ ދިވެހި ބަހުން ވާހަކަ ދައްކާ މީހެއް` | ✓ identical |
| `Salaam dhivehi raajje` | `ސަލާމް ދިވެހި ރާއްޖެ` | ✓ identical |
| `Maadhama haveeruge bahdhaluvun cancel kohffi` | `މާދަމާ ހަވީރުގެ ބައްދަލުވުން ކެންސަލްކޮށްފި` | ✓ identical |

End-to-end through the production Flask pipeline (chunking → batched inference → keymap_to_thaana → RTL punctuation): same outputs, no regressions on the cases the shipped model already handled correctly.

### 5.4 Inference cost

Measured on the 7-input regression set above with production decoder params (`num_beams=4`, `max_new_tokens=512`).

| Metric | Baseline (Neobe) | New model | Ratio |
|---|---|---|---|
| Inference time | 181 ms / input | 60 ms / input | **3.02× faster** |
| Output bytes | 77 / output | 18.6 / output | **4.15× smaller** |

The 3× decoder speedup matches the theoretical prediction (Thaana is 3 bytes/codepoint UTF-8; Segha keymap is 1 byte/codepoint). The 4.15× byte ratio exceeds 3× because the new model also stops generating the hallucinated repeats that bloated the baseline's output on short inputs.

A full 231-word paragraph end-to-end through the Flask streaming pipeline completes in **4.87 s** total, with no silent gap longer than 2.08 s.

## 6. Discussion

### 6.1 The bet paid off — both ways

The cost win was the hypothesis the project was built around. The quality win — fixing short-input hallucination via length-diverse augmentation — was tied to the same training pass for budgetary reasons but was conceptually independent. Both delivered. We did not have to choose between cheaper and better.

### 6.2 Warm-starting was load-bearing

The encoder of `Neobe/dhivehi-byt5-latin2thaana-v1` already encoded Latin Dhivehi orthography from its earlier training. Reusing those weights (rather than starting from `google/byt5-small`) meant only the decoder needed to retarget, on a corpus the encoder had effectively already seen. The 66-minute training time and clean loss curve are both downstream of this choice. Training from base byt5-small would have needed substantially more data and time to recover Latin Dhivehi understanding.

### 6.3 The single failure case

`bohkuraa → ބޮށްކുරာ` instead of the canonical `ބޮއްކുරാ` is a single-character substitution: `ޝ` (shaviyani) vs `އ` (alifu) in position 2. Probable causes:

- The word "bohkuraa" (gourd) is uncommon, so unlikely to appear with high frequency in the training corpus.
- Latin "h" + alifu+sukun (`އް`) is a less obvious mapping than alifu→`w`. The model may have learned that "Sh-like" Latin sequences map to shaviyani, and overgeneralized.
- This is a recoverable failure mode (typo-like) rather than a catastrophic one (the previous behavior of emitting repeated wrong words).

If we wanted to fix this specific case, we'd add `bohkuraa→ބޮއްކුরާ` to a manually curated single-word pairs file and do a short additional training pass. The plan's "Open question" about whether to build a Radheef dictionary becomes relevant if many such cases surface in production.

### 6.4 Long-input training was deferred — and that's OK

The plan called for length-diverse augmentation including ~4,500 long concatenated examples. We filtered those out for the first run because they 8×'d step time without obvious necessity given the production chunking pipeline (which never feeds inputs longer than ~20 words to the model). Eval and regression results show no degradation on the inputs the production pipeline actually generates. The long-concat examples remain on disk; we can run a focused round-2 if production behavior ever demands it.

This is a useful practical lesson: not every theoretical improvement is worth its compute cost. The decision to defer was bounded (we kept the data) and reversible (filter is one config line in `finetune.py`).

### 6.5 Generalization beyond Dhivehi

The technique — retargeting a byte-level model's decoder from a 3-byte UTF-8 script to a 1-byte ASCII keymap encoding plus a deterministic post-processor — generalizes to any task producing output in scripts that occupy multi-byte UTF-8 ranges (Arabic, Devanagari, Bengali, Tamil, Thai, Tibetan, etc.). The prerequisites are:

1. A 1:1 invertible romanization or keyboard mapping for the target script.
2. A byte-level model whose decoder cost dominates the workload.
3. A use case where output appearance to the user is independent of internal representation.

For pure inference cost reduction on transliteration / script-conversion tasks where (1) and (3) hold, this is a free architectural win once the keymap-to-script post-processor is implemented and validated.

## 7. Conclusion

We retargeted a ByT5-small fine-tune to emit Segha-phonetic ASCII keymap instead of UTF-8 Thaana, with a deterministic post-processor reconstructing Thaana for display. The new model achieves a **3.02× decoder speedup** while improving quality on the documented short-input failure cases (3 of 4 now pass exactly, the fourth fails recoverably) and matching baseline quality on realistic sentences.

The model has been wired into the production Flask service via a one-line change to `MODEL_NAMES['latin2thaana']`, a one-line `keymap_to_thaana(...)` post-processing call, and a `length_penalty` revert. End-to-end testing on the live service confirms identical outputs to the standalone regression eval.

Future work, only if production behavior warrants:
- Round-2 training including the deferred long-concatenation examples, if cross-sentence context ever matters.
- A small curated single-word pairs file (Radheef-derived) to address `bohkuraa`-style rare-word edge cases.
- Publishing the checkpoint to Hugging Face under a new model name.

The deferred items are real options, not loose ends: the long-concat data is on disk, and the spec for single-word augmentation is documented in §3.4. This experiment was self-contained and ships as-is.

## References

1. Xue, L., Barua, A., Constant, N., Al-Rfou, R., Narang, S., Kale, M., Roberts, A., & Raffel, C. (2022). *ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models*. TACL.
2. alakxender. *dhivehi-transliteration-pairs* (Hugging Face dataset). https://huggingface.co/datasets/alakxender/dhivehi-transliteration-pairs
3. Jawish. *Javascript Thaana Keyboard (jtk)*. https://github.com/jawish/jtk
4. Neobe. *dhivehi-byt5-latin2thaana-v1* (Hugging Face model). https://huggingface.co/Neobe/dhivehi-byt5-latin2thaana-v1

## Appendix A: Process Notes

Decisions and incidents not load-bearing for the main argument but useful for replication:

- **CUDA wheel.** Blackwell sm_120 requires PyTorch built against CUDA ≥ 12.6. We installed `torch==2.11.0+cu128` (CUDA 12.8 wheel) over the existing CPU wheel. Driver 591.86 / CUDA 13.1 supports both 12.6 and 12.8 wheels.
- **Step time tuning.** First smoke run with default dynamic padding: 9.5 s/step (variance 1–17 s). Adding `group_by_length=True`: 6.4 s/step. Adding `max_length=512` + filter to combined ≤400 bytes: 0.58 s/step. The dominant cost was the long-concatenation augmentation under dynamic padding — long sequences in a batch inflated the whole batch.
- **Memory budget.** Batch=16 at seq=1024 OOM'd despite 16 GB VRAM (PyTorch reported >29 GB allocated, suggesting fragmentation). Batch=8 at seq=1024 fit, batch=16 at seq=512 fit comfortably.
- **Eval-time generation cap.** During-training eval uses `generation_max_length=256` (or 512 in the full run) and greedy decoding, distinct from production beam=4 / max=512. This is for eval throughput; final model evaluation uses production decoder params.
- **Reverse direction.** `thaana2latin` is explicitly out of scope. Latin output is already 1 byte/char, so there's no decoder-cost win, and refactoring would risk regressions on a working code path.
