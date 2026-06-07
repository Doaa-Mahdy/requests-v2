FROM ollama/ollama:latest

# =========================
# System dependencies
# =========================
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    git \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# =========================
# Python dependencies
# =========================
COPY requirements.txt .

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# =========================
# Copy app
# =========================
COPY . .
# Preload models
RUN python app/scripts/download_models.py

# =========================
# Pre-pull model (BEST PRACTICE)
# =========================

RUN ollama serve & \
    sleep 10 && \
    ollama pull qwen2.5:7b

# 5. Environment & Entrypoint
ENV TRANSFORMERS_OFFLINE=1
ENV FRAUD_MODEL_PATH=/app/models/fraud_detection
ENV CLIP_MODEL_PATH=/app/models/clip_vit
ENV STT_MODEL_PATH=/app/models/stt_model
ENV VQA_MODEL_PATH=/app/models/qwen_vl

# Entrypoint setup
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]