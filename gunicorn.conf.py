# Gunicorn Configuration for Dhivehi Transliteration
import multiprocessing
import os

# Server socket
# PORT lets the host pick the port (HF Spaces, Cloud Run, ...); default 5001.
bind = f"0.0.0.0:{os.environ.get('PORT', '5001')}"
backlog = 2048

# Worker processes
# Each worker eager-loads both models (~2 GB RAM), so worker count is bounded by
# RAM, not cores. Override WEB_CONCURRENCY on small hosts — HF Spaces CPU Basic
# (2 vCPU) should run WEB_CONCURRENCY=1, GUNICORN_THREADS=4.
workers = int(os.environ.get("WEB_CONCURRENCY", 4))  # 4 workers = ~12 of 50 users each
worker_class = "gthread"  # Threaded workers for better concurrency
threads = int(os.environ.get("GUNICORN_THREADS", 2))  # 2 threads/worker = 8 concurrent
timeout = 180  # 3 minutes (model inference takes ~30-40s)
keepalive = 5

# Disable preload to avoid CUDA fork issues. Each worker imports app.py
# post-fork, and app.py eager-loads the model at import time.
preload_app = False

# Logging
accesslog = "-"  # stdout
errorlog = "-"   # stderr
loglevel = "info"

# Process naming
proc_name = "dhivehi-transliteration"

# Server mechanics
daemon = False
pidfile = None
umask = 0
user = None
group = None
tmp_upload_dir = None

# SSL (if needed later)
# keyfile = None
# certfile = None

print("\n" + "="*60)
print("Dhivehi Transliteration - Gunicorn Configuration")
print("="*60)
print(f"Workers: {workers}")
print(f"Worker class: {worker_class}")
print(f"Bind: {bind}")
print(f"Timeout: {timeout}s")
print(f"Preload app: {preload_app}")
print("="*60 + "\n")
