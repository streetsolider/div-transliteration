# Dhivehi Latin to Thaana Transliteration

Web application for converting Latin script Dhivehi text to Thaana script using the ByT5 model.

## Features

- Real-time streaming transliteration with progress updates
- Production-ready with Gunicorn (4 workers, handles 50+ concurrent users)
- Docker support for easy deployment
- Load testing tools included
- 500-word limit with live word counter
- Proper RTL punctuation (،؛؟)
- Systemd service for auto-start on Linux

## Quick Start (Development)

### Prerequisites
- Python 3.9+
- 8GB RAM minimum (model is ~1.5GB)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd div-transliteration

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run development server
python app.py
```

Visit http://localhost:5001

## Production Deployment

### Option 1: Docker (Recommended for containers)

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Option 2: Gunicorn (Recommended for Linux servers)

```bash
# Install dependencies
pip install -r requirements.txt

# Run with Gunicorn
gunicorn -c gunicorn.conf.py app:app
```

### Option 3: Systemd Service (Auto-start on boot)

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## Development

### Project Structure

```
div-transliteration/
├── app.py                          # Flask application
├── gunicorn.conf.py                # Production server config
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                  # Frontend UI
├── static/
│   └── fonts/
│       └── Faruma.ttf              # Thaana font
├── Dockerfile                      # Docker image
├── docker-compose.yml              # Docker orchestration
├── dhivehi-transliteration.service # Systemd service
├── load_test.py                    # Load testing script
├── DEPLOYMENT.md                   # Deployment guide
└── README.md                       # This file
```

### Running Tests

```bash
# Start the server first
gunicorn -c gunicorn.conf.py app:app

# Run load tests (in another terminal)
python load_test.py 10  # Test with 10 concurrent users
```

### Making Changes

1. **Edit model parameters** in `app.py`:
   - `num_beams`: Controls quality vs speed (1-8)
   - `repetition_penalty`: Reduces repetition (1.0-1.5)
   - `max_new_tokens`: Max output length

2. **Edit worker settings** in `gunicorn.conf.py`:
   - `workers`: Number of worker processes (4)
   - `threads`: Threads per worker (2)
   - `timeout`: Request timeout (180s)

3. **Edit frontend** in `templates/index.html`:
   - Word limit: Change `MAX_WORDS` constant
   - Styling: Edit CSS in `<style>` section

### Testing Changes

```bash
# Test locally first (faster than Docker)
gunicorn -c gunicorn.conf.py app:app

# Then rebuild Docker for deployment
docker-compose build
docker-compose up -d
```

## Configuration

### Gunicorn Settings

- **Workers**: 4 (for ~50 concurrent users)
- **Worker class**: gthread (threaded workers)
- **Threads per worker**: 2 (8 concurrent requests total)
- **Timeout**: 180 seconds
- **Preload app**: Enabled (model loads once, shared across workers)

### Model Settings

- **Model**: `Neobe/dhivehi-byt5-latin2thaana-v1`
- **Beam size**: Configurable (2-4 recommended)
- **Repetition penalty**: 1.0-1.3 recommended
- **Chunk size**: 20 words with 4-word overlap

## Performance

### Expected Response Times

- **Sequential**: ~7-27 seconds per request
- **Concurrent (10 users)**: ~22-56 seconds per request
- **Model loading**: ~8-10 seconds (first request only)

### Load Test Results

```
Sequential (5 users):
  100% success rate
  Average: 14s per request

Concurrent (10 users):
  100% success rate
  Total: 56s for all 10
  Average: 38.5s per request
```

## API Endpoint

### POST `/transliterate`

Transliterate Latin text to Thaana with SSE streaming.

**Request:**
```json
{
  "text": "salaam dhivehi raajje"
}
```

**Response:** Server-Sent Events (SSE) stream

```
data: {"status": "Starting...", "progress": 0}
data: {"status": "Processing...", "progress": 50, "thaana": "ސަލާމް", "partial": true}
data: {"status": "Complete!", "progress": 100, "thaana": "ސަލާމް ދިވެހި ރާއްޖެ", "partial": false}
```

## Troubleshooting

### Model won't load
- Check available RAM (needs 2GB minimum per worker)
- Reduce number of workers in `gunicorn.conf.py`

### Slow performance
- Reduce `num_beams` in `app.py` (faster but lower quality)
- Increase `workers` in `gunicorn.conf.py` (needs more RAM)
- Use native Gunicorn on Linux (faster than Docker on Mac)

### Port already in use
```bash
# Find process using port 5001
lsof -i :5001

# Kill it
kill -9 <PID>
```

## Tech Stack

- **Framework**: Flask 3.1.2
- **Server**: Gunicorn 21.2.0
- **Model**: Transformers 4.57.6 + PyTorch 2.8.0
- **Frontend**: Vanilla JavaScript with SSE
- **Deployment**: Docker, systemd

## Model Information

- **Name**: Neobe/dhivehi-byt5-latin2thaana-v1
- **Type**: ByT5 (Byte-level T5)
- **Task**: Text-to-text generation (transliteration)
- **Size**: ~1.5GB
- **License**: Check model card on Hugging Face

## Contributing

1. Create a feature branch from `development`
2. Make your changes
3. Test thoroughly (local + Docker)
4. Commit with descriptive message
5. Merge to `development`, then to `master`

## License

[Add your license here]

## Support

For deployment issues, see [DEPLOYMENT.md](DEPLOYMENT.md)

For bugs or feature requests, open an issue in the repository.
