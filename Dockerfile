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

# Copy application code
COPY app.py .
COPY gunicorn.conf.py .
COPY templates/ templates/
COPY static/ static/

# Expose application port
EXPOSE 5001

# Run Gunicorn
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
