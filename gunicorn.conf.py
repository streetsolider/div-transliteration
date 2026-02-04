# Gunicorn Configuration for Dhivehi Transliteration
import multiprocessing

# Server socket
bind = "0.0.0.0:5001"
backlog = 2048

# Worker processes
workers = 4  # For 50 users, 4 workers = ~12 users per worker
worker_class = "gthread"  # Threaded workers for better concurrency
threads = 2  # 2 threads per worker = 8 concurrent requests total
timeout = 180  # 3 minutes (model inference takes ~30-40s)
keepalive = 5

# Preload app to load model before forking workers
# This way model loads once, then workers are forked with model in memory
preload_app = True

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
