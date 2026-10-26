# Multi-stage production container for ProctorAI
FROM python:3.10-slim AS base

# Install system audio, video, and OpenGL libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install python dependencies without root cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Expose Streamlit default HTTP port
EXPOSE 8501

# Configure container health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Execute Streamlit server
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
