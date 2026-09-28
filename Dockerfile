# ==============================================================================
# Dockerfile: PassportReadAPI
# Passport MRZ reading and bio-data parsing service
# ==============================================================================

# Use official Python 3.11 slim image
FROM python:3.11-slim

# Prevent bytecode caching and ensure immediate log output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Set container working directory
WORKDIR /app

# Install system dependencies required for image parsing and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose internal listening port for PassportReadAPI
EXPOSE 8002

# Start server using uvicorn (adjust if your entrypoint is app.py -> app:app)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
