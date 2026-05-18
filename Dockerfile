# Dhivehi Transliteration - Production Docker Image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download both models during build so they're cached in the image.
# latin2thaana: keymap-output ByT5 (~3x decoder speedup; see README.md).
# thaana2latin: original Neobe checkpoint.
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    [AutoTokenizer.from_pretrained(n) and AutoModelForSeq2SeqLM.from_pretrained(n) \
     for n in ['str33t/dhivehi-byt5-latin2thaana-keymap-v1', \
               'Neobe/dhivehi-byt5-thaana2latin-v1']]"

# Copy application code
COPY app.py .
COPY keymap.py .
COPY gunicorn.conf.py .
COPY templates/ templates/
COPY static/ static/

# Expose application port
EXPOSE 5001

# Run Gunicorn
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
