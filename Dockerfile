# Dhivehi Transliteration - Production Docker Image
#
# Doubles as the Hugging Face Spaces image (`sdk: docker`). Spaces runs the
# container as UID 1000, so the user is created before any pip install or model
# download — otherwise the baked-in cache is unreadable at runtime and the app
# re-downloads both checkpoints on every cold start.
FROM python:3.9-slim

# Install system dependencies for building Python packages (needs root)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set up the UID-1000 user Spaces expects, then do everything else as that user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# torch build variant. `cpu` skips the bundled CUDA libraries and keeps the
# image ~2.5 GB smaller, which matters for Spaces build time and cold starts.
# GPU builds pass --build-arg TORCH_VARIANT=cu128 (see docker-compose-gpu.yml).
ARG TORCH_VARIANT=cpu

# Copy requirements first (Docker layer caching)
COPY --chown=user requirements.txt .

# pip must be upgraded first: the 23.x bundled in python:3.9-slim rejects every
# wheel on the PyTorch index whose filename uses underscores (typing_extensions)
# as an "inconsistent Name", which makes resolution fail outright.
#
# Then install torch from the variant-specific index; the torch==2.8.0 pin in
# requirements.txt is satisfied by 2.8.0+cpu and is not re-downloaded.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.8.0 \
        --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    && pip install --no-cache-dir -r requirements.txt

# Pre-download both models during build so they're cached in the image.
# latin2thaana: keymap-output ByT5 (~3x decoder speedup; see README.md).
# thaana2latin: original Neobe checkpoint.
RUN python -c "from transformers import AutoTokenizer, AutoModelForSeq2SeqLM; \
    [AutoTokenizer.from_pretrained(n) and AutoModelForSeq2SeqLM.from_pretrained(n) \
     for n in ['str33t/dhivehi-byt5-latin2thaana-keymap-v1', \
               'Neobe/dhivehi-byt5-thaana2latin-v1']]"

# Copy application code
COPY --chown=user app.py .
COPY --chown=user keymap.py .
COPY --chown=user gunicorn.conf.py .
COPY --chown=user templates/ templates/
COPY --chown=user static/ static/

# Expose application port (matches `app_port: 5001` in the Space README)
EXPOSE 5001

# Run Gunicorn
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
