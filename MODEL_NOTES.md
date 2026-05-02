# Model Limitations & Notes for Future Training

Notes on observed limitations of `Neobe/dhivehi-byt5-latin2thaana-v1` (and to a lesser extent `Neobe/dhivehi-byt5-thaana2latin-v1`) and why a model-level fix (fine-tuning on augmented data) is the right path forward instead of further decoder tuning.

## Known issue: short-input repetition / hallucination

Single-word and very short Latin inputs produce repetitive or hallucinated multi-word output. The model treats short ambiguous inputs as sentence prefixes from news headlines and "completes" them.

### Concrete failing cases (with current shipped params: `length_penalty=1.2, repetition_penalty=1.0, early_stopping=False`)

| Input | Expected | Actual |
|---|---|---|
| `bohkuraa` | `ބޮއްކުރާ` | `ބޮއްކުރާ ބޮއްކުރާ ބޮއްކުރާ ބޮއްކުރާ` (4×) |
| `kuru` (= "short") | `ކުރު` | `ކުރުމުގެ މައްސަލަ ކުރު ކުރުމުގެ ކުރުމުގެ ކުރުވާލައިފި` |
| `karu` (= "throat") | `ކަރު` | `ކަރުގެ ކަރުގެ ކަރުގެ ކަރު ކަރު ކަރު ކަރު` |
| `bas` (= "language" / "bus") | `ބަސް` or `ބާސް` | `ބާސްގެ ބާސް ކުރިއަށް، ބާސް ބާސް ބާސް` |

`bohkuraa` is pure repetition (beam search exploits `length_penalty>1` to fill `max_new_tokens=512`). `kuru/karu/bas` are worse — the model emits *related but different* word forms (e.g. "ކުރުމުގެ" = "of doing", "ކަރުގެ" = "of the neck/throat") because in its training distribution those Latin strings only appear as the start of longer phrases.

## Why decoder tuning is not the fix

We swept `length_penalty ∈ {-3, -2, -1, 0, 0.6, 0.8, 1.0, 1.2}`, `repetition_penalty ∈ {1.0, 1.2, 1.5, 1.7, 2.0, 2.5, 3.0}`, `no_repeat_ngram_size ∈ {4, 6, 8, 10, 12}`, and `early_stopping ∈ {True, False}`. **No combination simultaneously fixes short inputs without harming longer chunks.**

| Setting | Helps short inputs | Cost on longer inputs |
|---|---|---|
| `length_penalty=-1.0` | Fixes `bohkuraa` cleanly | None observed on chunk-sized inputs (≤16 words) |
| `repetition_penalty=2.0` | Fixes `bas` cleanly | **Drops real words** — e.g. "Aharennakee" disappears from "Aharennakee Dhivehi bahun..." because the model thinks the "Aharen" prefix is a repetition of an earlier "Aharen" elsewhere in the sentence |
| `repetition_penalty=1.5` + word-count trim | Fixes most short inputs (kuru, karu, bas) | Doesn't fix all ambiguous inputs; trim is a heuristic, not a real fix; model still emits the *wrong* word for `kuru` (returns "ކުރުމުގެ" = "of doing", not "ކުރު" = "short") |
| `no_repeat_ngram_size=8` or smaller | Helps short inputs | Mangles real text (corrupts spellings, replaces words mid-sentence) |
| `length_penalty=-2.0` or stronger | No additional help | Truncates legitimate longer outputs |

The fundamental tension: the model genuinely scores "word + word + EOS" higher than "word + EOS" for short inputs because that's the distribution it was trained on. Decoder params can suppress this for *some* words but not all — e.g. `karu` keeps appending headline continuations (`ކަރުގެ މިނިސްޓަރު` = "of the throat minister") under any rep_penalty we tried.

## Workarounds we considered and rejected

These live in `git stash@{0}` (message: "decoder param + word-count trim attempts for short-input repetition (model-level fix needed)") if you want to look at them:

1. `length_penalty=-1.0`, `repetition_penalty=1.5`, `early_stopping=True` in `model.generate(...)`.
2. Post-processing trim: for chunks with 1–3 input words, cap output word count at the input word count.

The post-processing trim works for the specific failing cases but is brittle — it relies on word-count parity that may not hold for all inputs (e.g. multi-word abbreviations expanded into a single Thaana phrase, or vice versa).

## Path forward: fine-tune with augmented data

The robust fix is at the model level. Suggested data augmentation for the next training run:

- **Single-word pairs**: pair every common standalone Dhivehi word with its Latin transliteration. Sources: a Dhivehi dictionary, Radheef, frequency-sorted word list.
- **Short 2–3 word phrase pairs**: common short utterances ("varah ragalhu", "kihinehtha", "shukuriyya", etc.) without surrounding context.
- **Specifically include the failing cases above**: `kuru → ކުރު`, `karu → ކަރު`, `bas → ބަސް`, `bohkuraa → ބޮއްކުރާ`, plus other short ambiguous Latin forms.
- **Balance against the existing news-text corpus** so the model retains its strong long-form behavior. A 5–10% augmentation by short-input pairs is likely sufficient.

After fine-tuning, decoder params can probably revert to defaults (or `length_penalty=1.0, repetition_penalty=1.0`) since the underlying distribution will support short inputs natively.

## Reverse direction (`thaana2latin`) shares the params

Both directions use the same `model.generate()` call in `app.py`. Whatever decoder settings or model swap we apply to `latin2thaana` should be applied identically to `thaana2latin`. The reverse direction is faster (Latin output ≈ half the bytes of Thaana output for ByT5) and showed similar — though less severe — short-input issues in spot checks.

## Reference: where the params live

- Decoder params: `app.py` inside the `generate()` definition, in the `model.generate(...)` call.
- Chunking pipeline: same function — splits paragraphs → sentences (`[.!?]`) → phrases (`[,;]`) → 20-word windows. Do not change these regexes when retraining.
- Models: `MODEL_NAMES` dict at the top of `app.py`. Update names there after publishing fine-tuned weights to Hugging Face.

_Last reviewed: 2026-05-02._
