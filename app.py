from flask import Flask, render_template, request, jsonify, Response
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import json
import re

app = Flask(__name__)

# Load model and tokenizer for streaming support
print("Loading ByT5 model...")
model_name = "Neobe/dhivehi-byt5-latin2thaana-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
print("Model loaded successfully!")

# Store active generations
active_generations = {}

@app.route('/')
def index():
    return render_template('index.html')

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
            # Send initial status
            yield f"data: {json.dumps({'status': 'Starting...', 'request_id': request_id})}\n\n"

            # Store that this generation is active
            active_generations[request_id] = True

            # Split text into paragraphs first (preserve paragraph breaks)
            paragraphs = text.split('\n\n')
            all_paragraphs_thaana = []

            # Helper function to split text into chunks of max N words
            def split_into_word_chunks(text, max_words=20):
                words = text.split()
                chunks = []
                for i in range(0, len(words), max_words):
                    chunks.append(' '.join(words[i:i + max_words]))
                return chunks

            # Process each paragraph
            for para_idx, paragraph in enumerate(paragraphs):
                if not paragraph.strip():
                    # Empty paragraph, preserve the break
                    all_paragraphs_thaana.append('')
                    continue

                # Split paragraph into sentences while preserving punctuation
                sentence_pattern = r'[^.!?]+[.!?]+|[^.!?]+$'
                sentences = re.findall(sentence_pattern, paragraph)
                sentences = [s.strip() for s in sentences if s.strip()]

                # If no sentences found, treat entire paragraph as one sentence
                if not sentences:
                    sentences = [paragraph]

                # Process each sentence with phrase-level chunking
                all_thaana = []
                total_sentences = len(sentences)

                for sent_idx, sentence in enumerate(sentences, 1):
                    if request_id not in active_generations:
                        # Generation was stopped
                        yield f"data: {json.dumps({'status': 'Stopped', 'thaana': ' '.join(all_thaana), 'partial': True})}\n\n"
                        return

                    # Extract sentence-ending punctuation
                    ending_punct = ''
                    sentence_text = sentence
                    if sentence and sentence[-1] in '.!?':
                        ending_punct = sentence[-1]
                        sentence_text = sentence[:-1].strip()

                    # Split sentence into phrases by commas and semicolons
                    phrase_pattern = r'[^,;]+[,;]?'
                    phrases = re.findall(phrase_pattern, sentence_text)
                    phrases = [p.strip() for p in phrases if p.strip()]

                    # Process each phrase
                    sentence_thaana_parts = []

                    for phrase in phrases:
                        # Extract phrase delimiter (comma or semicolon)
                        phrase_delimiter = ''
                        phrase_text = phrase
                        if phrase and phrase[-1] in ',;':
                            phrase_delimiter = phrase[-1]
                            phrase_text = phrase[:-1].strip()

                        # Split phrase into word chunks if too long
                        word_count = len(phrase_text.split())
                        if word_count > 20:
                            chunks = split_into_word_chunks(phrase_text, max_words=20)
                        else:
                            chunks = [phrase_text]

                        # Process each chunk
                        phrase_thaana_parts = []
                        total_chunks = len(chunks)

                        for chunk_idx, chunk in enumerate(chunks, 1):
                            if request_id not in active_generations:
                                yield f"data: {json.dumps({'status': 'Stopped', 'thaana': ' '.join(all_thaana), 'partial': True})}\n\n"
                                return

                            # Update status
                            if total_chunks > 1:
                                status_msg = f'Sentence {sent_idx}/{total_sentences}, chunk {chunk_idx}/{total_chunks}...'
                            else:
                                status_msg = f'Processing sentence {sent_idx}/{total_sentences}...'
                            yield f"data: {json.dumps({'status': status_msg, 'request_id': request_id})}\n\n"

                            # Tokenize and generate
                            inputs = tokenizer(chunk, return_tensors="pt", truncation=False, padding=False)
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=512,
                                num_beams=4,
                                do_sample=False,
                                early_stopping=False,
                                length_penalty=1.2,
                                repetition_penalty=1.0,
                            )

                            # Decode chunk
                            chunk_thaana = tokenizer.decode(outputs[0], skip_special_tokens=True)
                            phrase_thaana_parts.append(chunk_thaana)

                        # Rejoin chunks and add phrase delimiter
                        phrase_thaana = ' '.join(phrase_thaana_parts)
                        if phrase_delimiter:
                            phrase_thaana += phrase_delimiter
                        sentence_thaana_parts.append(phrase_thaana)

                    # Rejoin all phrases and add sentence-ending punctuation
                    sentence_thaana = ' '.join(sentence_thaana_parts)
                    if ending_punct:
                        sentence_thaana += ending_punct

                    # Replace LTR punctuation with RTL equivalents
                    sentence_thaana = sentence_thaana.replace(',', '،')  # Arabic comma
                    sentence_thaana = sentence_thaana.replace(';', '؛')  # Arabic semicolon

                    all_thaana.append(sentence_thaana)

                    # Send partial result with all completed paragraphs + current progress
                    current_paragraph_progress = ' '.join(all_thaana)
                    all_progress = all_paragraphs_thaana + [current_paragraph_progress]
                    partial_result = '\n\n'.join(all_progress)
                    yield f"data: {json.dumps({'status': f'Sentence {sent_idx}/{total_sentences} complete', 'thaana': partial_result, 'partial': True})}\n\n"

                # After processing all sentences in the paragraph, join them
                paragraph_thaana = ' '.join(all_thaana)
                all_paragraphs_thaana.append(paragraph_thaana)

            # Join all paragraphs with double newlines (preserve paragraph breaks)
            final_thaana = '\n\n'.join(all_paragraphs_thaana)
            yield f"data: {json.dumps({'status': 'Complete!', 'thaana': final_thaana, 'latin': text, 'partial': False})}\n\n"

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

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌐 Dhivehi Transliteration Web App")
    print("="*60)
    print("📍 Open your browser and go to: http://localhost:5001")
    print("Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    app.run(debug=True, port=5001)
