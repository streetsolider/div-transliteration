from flask import Flask, render_template, request, jsonify, Response
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import time
import json
import re

app = Flask(__name__)

# Lazy-loaded model (initialized per worker to avoid CUDA fork issues)
model_name = "Neobe/dhivehi-byt5-latin2thaana-v1"
device = None
tokenizer = None
model = None

def get_model():
    """Load model on first use (lazy init to avoid CUDA fork issues with Gunicorn)."""
    global device, tokenizer, model
    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print("Loading ByT5 model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        print("Model loaded successfully!")
    return device, tokenizer, model

# Store active generations
active_generations = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ready')
def ready():
    if model is not None:
        return jsonify({'status': 'ready'}), 200
    return jsonify({'status': 'loading'}), 503

@app.route('/transliterate', methods=['POST'])
def transliterate():
    """API endpoint - uses working pipeline method"""
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    # Generate request ID
    request_id = str(time.time())

    def generate():
        try:
            # Model is eager-loaded at worker startup (see gunicorn post_worker_init);
            # this is just a safety fallback if someone runs app.py directly.
            device, tokenizer, model = get_model()

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

            # Helper function to split text into overlapping chunks
            def split_into_word_chunks(text, max_words=20, overlap=4):
                """Split text into overlapping chunks for context preservation.

                Args:
                    text: Input text to chunk
                    max_words: Maximum words per chunk (default 20)
                    overlap: Number of words to overlap between chunks (default 4 = 20%)

                Returns:
                    List of (chunk_text, is_first_chunk) tuples
                """
                words = text.split()
                chunks = []
                stride = max_words - overlap  # 20 - 4 = 16 words stride

                for i in range(0, len(words), stride):
                    chunk_words = words[i:i + max_words]
                    chunk_text = ' '.join(chunk_words)
                    is_first_chunk = (i == 0)
                    chunks.append((chunk_text, is_first_chunk))

                return chunks

            # Process each paragraph — collect all chunks and run ONE batched
            # generate() call per paragraph. Much faster than per-chunk sequential
            # inference for multi-chunk inputs, at the cost of losing sentence-level
            # streaming (user sees progress per paragraph, not per sentence).
            overlap_words = 4  # must match split_into_word_chunks overlap

            for para_idx, paragraph in enumerate(paragraphs):
                if not paragraph.strip():
                    all_paragraphs_thaana.append('')
                    continue

                if request_id not in active_generations:
                    yield f"data: {json.dumps({'status': 'Stopped', 'thaana': '\\n\\n'.join(all_paragraphs_thaana), 'partial': True})}\n\n"
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

                        if len(phrase_text.split()) > 20:
                            chunks = split_into_word_chunks(phrase_text, max_words=20)
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
                status_msg = (
                    f'Processing paragraph {para_idx + 1}/{len(paragraphs)} '
                    f'({len(flat_chunks)} chunk{"s" if len(flat_chunks) != 1 else ""} batched)...'
                )
                yield f"data: {json.dumps({'status': status_msg, 'request_id': request_id, 'progress': progress})}\n\n"

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
                        length_penalty=1.2,
                        repetition_penalty=1.0,
                    )
                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

                # Stitch decoded chunks back into sentences using the plan
                all_thaana = []
                for sent_plan in plan:
                    sentence_thaana_parts = []
                    for phrase_plan in sent_plan['phrases']:
                        phrase_chunks = []
                        for i in range(phrase_plan['start'], phrase_plan['end']):
                            chunk_thaana = decoded[i]
                            if not flat_is_first[i] and phrase_plan['multi']:
                                output_words = chunk_thaana.split()
                                if len(output_words) > overlap_words:
                                    chunk_thaana = ' '.join(output_words[overlap_words:])
                            phrase_chunks.append(chunk_thaana)
                        phrase_thaana = ' '.join(phrase_chunks)
                        if phrase_plan['delimiter']:
                            phrase_thaana += phrase_plan['delimiter']
                        sentence_thaana_parts.append(phrase_thaana)

                    sentence_thaana = ' '.join(sentence_thaana_parts)
                    if sent_plan['ending_punct']:
                        sentence_thaana += sent_plan['ending_punct']

                    # Replace LTR punctuation with RTL equivalents
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
                yield f"data: {json.dumps({'status': f'Paragraph {para_idx + 1}/{len(paragraphs)} complete', 'thaana': partial_result, 'partial': True, 'progress': progress})}\n\n"

            # Join all paragraphs with double newlines (preserve paragraph breaks)
            final_thaana = '\n\n'.join(all_paragraphs_thaana)
            yield f"data: {json.dumps({'status': 'Complete!', 'thaana': final_thaana, 'latin': text, 'partial': False, 'progress': 100})}\n\n"

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

# Eager-load at import time so Flask dev (`flask run` / `python app.py`) and
# gunicorn workers (with preload_app=False, import happens post-fork) both
# have the model ready before serving any request. Idempotent — guarded by
# the `if model is None` check inside get_model().
get_model()

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Dhivehi Transliteration Web App")
    print("="*60)
    print("Open your browser and go to: http://localhost:5001")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5001)
