from flask import Flask, render_template, request, jsonify, Response
from werkzeug.middleware.proxy_fix import ProxyFix
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import os
import time
import json
import re

from keymap import keymap_to_thaana

app = Flask(__name__)
# The page builds every URL it fetches (/ready, /transliterate, the fonts) with
# url_for, so it must know the path the proxy mounts us under. ProxyFix reads
# X-Forwarded-Prefix (Traefik's default, nginx via proxy_set_header) into
# SCRIPT_NAME; a proxy that sends nothing can set the SCRIPT_NAME env var
# instead, which gunicorn passes through. Without this, a deployment under
# /anything/ gets 404s for the fonts and a ready poll that never returns 200,
# so the "Loading AI model" overlay never clears.
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# latin2thaana: fine-tune that emits Segha-phonetic ASCII keymap; we convert
# keymap -> Thaana via keymap_to_thaana after decode (~3x decoder speedup vs.
# emitting Thaana UTF-8 directly). See README.md.
# Lazy-loaded models (initialized per worker to avoid CUDA fork issues)
MODEL_NAMES = {
    'latin2thaana': "str33t/dhivehi-byt5-latin2thaana-keymap-v1",
    'thaana2latin': "Neobe/dhivehi-byt5-thaana2latin-v1",
}
device = None
models = {}  # direction -> {'tokenizer': ..., 'model': ...}

def get_model(direction='latin2thaana'):
    """Load model on first use (lazy init to avoid CUDA fork issues with Gunicorn)."""
    global device
    if direction not in MODEL_NAMES:
        raise ValueError(f"Unknown direction: {direction}")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    if direction not in models:
        name = MODEL_NAMES[direction]
        print(f"Loading ByT5 model: {name}")
        tok = AutoTokenizer.from_pretrained(name)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(name).to(device)
        models[direction] = {'tokenizer': tok, 'model': mdl}
        print(f"Loaded: {name}")
    entry = models[direction]
    return device, entry['tokenizer'], entry['model']

# 10-word chunks with no overlap, chosen from measurement rather than taste.
#
# At 20 words the model silently drops content — on a real news paragraph it
# rendered "thulhadhoo dhaairage MP Abdul Hannan sponsor koh" as just
# "ތުޅަދޫ ދާއިރާގެ ސްޕޮންސަރުކޮށް", losing the member's name entirely (20 words in,
# 15 out). That matches the long-input degradation recorded in MODEL_NOTES.md.
#
# The 4-word overlap was meant to protect chunk boundaries, but it cannot be
# undone reliably: the model does not preserve word counts, and it transliterates
# the same overlapped words differently in each chunk (މަޖިލީހު vs މަޖިލިސް). The
# old fixed trim — drop exactly `overlap` output words from every non-first chunk
# — therefore deleted real content whenever the overlap region compressed, which
# is how words went missing mid-paragraph. At 10 words the model stops dropping
# content, so the boundary protection is no longer needed and the whole class of
# stitching error goes away.
#
# Kept at module scope (not nested in the SSE generator) so it is testable —
# see tests/test_chunking.py.
CHUNK_WORDS = 10
OVERLAP_WORDS = 0


def split_into_word_chunks(text, max_words=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Split a long phrase into chunks the model handles without dropping words.

    With OVERLAP_WORDS at 0 these are contiguous, so stitching is a plain
    concatenation. The overlap parameter is kept because the stride logic still
    reads it, and the guard below still matters if it is ever raised again:
    without it the stride could emit a final chunk wholly contained in the
    previous one (35 words at stride 16 gave a 3-word chunk covering words
    32-34, already inside the 16-34 chunk), which was then too short to trim and
    got duplicated in the output.

    Returns a list of (chunk_text, is_first_chunk) tuples.
    """
    words = text.split()
    stride = max_words - overlap
    chunks = []
    for i in range(0, len(words), stride):
        chunks.append((' '.join(words[i:i + max_words]), i == 0))
        if i + max_words >= len(words):
            break
    return chunks


def drop_repeated_prefix(prev, cur, max_overlap=OVERLAP_WORDS):
    """Remove the part of `cur` that repeats the tail of `prev`.

    A no-op while OVERLAP_WORDS is 0. If the overlap is ever raised again, this
    finds the longest *actual* repeat (up to max_overlap words) and drops exactly
    that, instead of assuming N input words came back as N output words. If
    nothing matches it drops nothing: a duplicated word is visible and harmless,
    a deleted one is silent.
    """
    prev_words, cur_words = prev.split(), cur.split()
    for k in range(min(max_overlap, len(prev_words), len(cur_words)), 0, -1):
        if prev_words[-k:] == cur_words[:k]:
            return ' '.join(cur_words[k:])
    return cur


# Store active generations
active_generations = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ready')
def ready():
    if all(d in models for d in MODEL_NAMES):
        return jsonify({'status': 'ready'}), 200
    return jsonify({'status': 'loading'}), 503

@app.route('/transliterate', methods=['POST'])
def transliterate():
    """API endpoint - uses working pipeline method"""
    data = request.get_json()
    text = data.get('text', '')
    direction = data.get('direction', 'latin2thaana')

    if not text:
        return jsonify({'error': 'No text provided'}), 400
    if direction not in MODEL_NAMES:
        return jsonify({'error': f'Invalid direction: {direction}'}), 400

    # Generate request ID
    request_id = str(time.time())

    def generate():
        try:
            # Model is eager-loaded at worker startup (see gunicorn post_worker_init);
            # this is just a safety fallback if someone runs app.py directly.
            device, tokenizer, model = get_model(direction)

            # Store that this generation is active
            active_generations[request_id] = True

            # Split text into paragraphs first (preserve paragraph breaks)
            paragraphs = text.split('\n\n')
            all_paragraphs_thaana = []

            # Calculate total sentences for progress tracking
            total_sentences_all = 0
            for paragraph in paragraphs:
                if paragraph.strip():
                    sentence_pattern = r'[^.!?]+[.!?]+|[^.!?]+$'
                    sentences = re.findall(sentence_pattern, paragraph)
                    sentences = [s.strip() for s in sentences if s.strip()]
                    if not sentences:
                        sentences = [paragraph]
                    total_sentences_all += len(sentences)

            completed_sentences = 0

            # Process each paragraph — collect all chunks and run ONE batched
            # generate() call per paragraph. Much faster than per-chunk sequential
            # inference for multi-chunk inputs, at the cost of losing sentence-level
            # streaming (user sees progress per paragraph, not per sentence).
            # Batching stays per-paragraph so the SSE stream can emit a partial
            # result as each paragraph finishes.
            for para_idx, paragraph in enumerate(paragraphs):
                if not paragraph.strip():
                    all_paragraphs_thaana.append('')
                    continue

                if request_id not in active_generations:
                    # Joined outside the f-string: a backslash inside an f-string
                    # expression is a SyntaxError before Python 3.12 (PEP 701).
                    stopped_thaana = '\n\n'.join(all_paragraphs_thaana)
                    yield f"data: {json.dumps({'status': 'Stopped', 'thaana': stopped_thaana, 'partial': True})}\n\n"
                    return

                # Split paragraph into sentences while preserving punctuation
                sentence_pattern = r'[^.!?]+[.!?]+|[^.!?]+$'
                sentences = re.findall(sentence_pattern, paragraph)
                sentences = [s.strip() for s in sentences if s.strip()]
                if not sentences:
                    sentences = [paragraph]

                # Build a plan of (sentences → phrases → chunk indices) and
                # flatten all chunks into a single list for one batched call.
                flat_chunks = []       # chunk text fed to tokenizer
                flat_is_first = []     # parallel: is_first_chunk flag for overlap trimming
                plan = []              # list of sentence descriptors

                for sentence in sentences:
                    ending_punct = ''
                    sentence_text = sentence
                    if sentence and sentence[-1] in '.!?':
                        ending_punct = sentence[-1]
                        sentence_text = sentence[:-1].strip()

                    phrase_pattern = r'[^,;]+[,;]?'
                    phrases = re.findall(phrase_pattern, sentence_text)
                    phrases = [p.strip() for p in phrases if p.strip()]

                    phrase_plans = []
                    for phrase in phrases:
                        phrase_delimiter = ''
                        phrase_text = phrase
                        if phrase and phrase[-1] in ',;':
                            phrase_delimiter = phrase[-1]
                            phrase_text = phrase[:-1].strip()

                        if len(phrase_text.split()) > CHUNK_WORDS:
                            chunks = split_into_word_chunks(phrase_text)
                        else:
                            chunks = [(phrase_text, True)]

                        start = len(flat_chunks)
                        for chunk_text, is_first in chunks:
                            flat_chunks.append(chunk_text)
                            flat_is_first.append(is_first)
                        end = len(flat_chunks)

                        phrase_plans.append({
                            'delimiter': phrase_delimiter,
                            'start': start,
                            'end': end,
                            'multi': (end - start) > 1,
                        })

                    plan.append({'ending_punct': ending_punct, 'phrases': phrase_plans})

                if not flat_chunks:
                    all_paragraphs_thaana.append('')
                    continue

                # Status update before batched inference
                progress = int((completed_sentences / total_sentences_all) * 100) if total_sentences_all > 0 else 0
                yield f"data: {json.dumps({'status': f'{progress}%', 'request_id': request_id, 'progress': progress})}\n\n"

                # Single batched forward pass for the entire paragraph
                inputs = tokenizer(
                    flat_chunks,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                ).to(device)
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        num_beams=4,
                        do_sample=False,
                        early_stopping=False,
                        length_penalty=1.0,
                        repetition_penalty=1.0,
                    )
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                # latin2thaana model now emits Segha-phonetic keymap; convert
                # to Thaana before downstream stitching.
                if direction == 'latin2thaana':
                    decoded = [keymap_to_thaana(s) for s in decoded]

                # Stitch decoded chunks back into sentences using the plan
                all_thaana = []
                for sent_plan in plan:
                    sentence_thaana_parts = []
                    for phrase_plan in sent_plan['phrases']:
                        phrase_chunks = []
                        for i in range(phrase_plan['start'], phrase_plan['end']):
                            chunk_thaana = decoded[i]
                            # Non-first chunks would repeat the previous chunk's
                            # tail if OVERLAP_WORDS were raised; a no-op at 0.
                            if not flat_is_first[i] and phrase_plan['multi'] and phrase_chunks:
                                chunk_thaana = drop_repeated_prefix(
                                    phrase_chunks[-1], chunk_thaana
                                )
                            phrase_chunks.append(chunk_thaana)
                        phrase_thaana = ' '.join(phrase_chunks)
                        if phrase_plan['delimiter']:
                            phrase_thaana += phrase_plan['delimiter']
                        sentence_thaana_parts.append(phrase_thaana)

                    sentence_thaana = ' '.join(sentence_thaana_parts)
                    if sent_plan['ending_punct']:
                        sentence_thaana += sent_plan['ending_punct']

                    # Replace LTR punctuation with RTL equivalents (Thaana output only)
                    if direction == 'latin2thaana':
                        sentence_thaana = sentence_thaana.replace(',', '،')
                        sentence_thaana = sentence_thaana.replace(';', '؛')
                        sentence_thaana = sentence_thaana.replace('?', '؟')

                    all_thaana.append(sentence_thaana)
                    completed_sentences += 1

                paragraph_thaana = ' '.join(all_thaana)
                all_paragraphs_thaana.append(paragraph_thaana)

                # Paragraph-level partial result for streaming UX
                progress = int((completed_sentences / total_sentences_all) * 100) if total_sentences_all > 0 else 0
                partial_result = '\n\n'.join(all_paragraphs_thaana)
                yield f"data: {json.dumps({'status': f'{progress}%', 'thaana': partial_result, 'partial': True, 'progress': progress})}\n\n"

            # Join all paragraphs with double newlines (preserve paragraph breaks)
            final_thaana = '\n\n'.join(all_paragraphs_thaana)
            yield f"data: {json.dumps({'status': '100%', 'thaana': final_thaana, 'latin': text, 'partial': False, 'progress': 100})}\n\n"

            # Cleanup
            if request_id in active_generations:
                del active_generations[request_id]

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            if request_id in active_generations:
                del active_generations[request_id]

    return Response(generate(), mimetype='text/event-stream')

@app.route('/stop/<request_id>', methods=['POST'])
def stop_generation(request_id):
    """Stop an active generation"""
    if request_id in active_generations:
        del active_generations[request_id]
        return jsonify({'status': 'stopped'})
    return jsonify({'status': 'not_found'}), 404

# Eager-load both directions at import time so Flask dev (`flask run` /
# `python app.py`) and gunicorn workers (with preload_app=False, import happens
# post-fork) both have models ready before serving any request. Idempotent —
# guarded by the `if direction not in models` check inside get_model().
#
# SKIP_MODEL_PRELOAD lets tests import the module for its pure text-shaping
# helpers without pulling 2.3 GB of weights. Never set it when serving.
if not os.environ.get('SKIP_MODEL_PRELOAD'):
    for _direction in MODEL_NAMES:
        get_model(_direction)

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Dhivehi Transliteration Web App")
    print("="*60)
    print("Open your browser and go to: http://localhost:5001")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5001)
