# Start from official Ollama image
FROM ollama/ollama:latest

# Install system dependencies
# These are required to compile C++ extensions for faiss, opencv, and torch
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy your scripts and cache models
COPY app/scripts/ /app/scripts/
RUN python3 /app/scripts/download_models.py

# Copy the rest of your application
COPY . .

# Set environment variables
ENV TRANSFORMERS_OFFLINE=1
ENV FRAUD_MODEL_PATH=/app/models/fraud_detection
ENV CLIP_MODEL_PATH=/app/models/clip_vit
ENV STT_MODEL_PATH=/app/models/stt_model
ENV VQA_MODEL_PATH=/app/models/qwen_vl

# Entrypoint setup
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]