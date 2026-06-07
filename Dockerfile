FROM ollama/ollama:latest

# =========================
# System dependencies
# =========================
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
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
# Create virtual environment (FIX)
# =========================
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# =========================
# Python dependencies
# =========================
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# =========================
# Copy app
# =========================
COPY . .

# =========================
# Pre-download python models (safe)
# =========================
RUN python app/scripts/download_models.py

# =========================
# Ollama model preload (IMPORTANT FIX)
# =========================
RUN ollama serve >/tmp/ollama.log 2>&1 & \
    sleep 15 && \
    ollama pull qwen2.5:7b && \
    pkill ollama || true

# =========================
# Environment
# =========================
ENV TRANSFORMERS_OFFLINE=1
ENV OLLAMA_HOST=http://localhost:11434
ENV FRAUD_MODEL_PATH=/app/models/fraud_detection
ENV CLIP_MODEL_PATH=/app/models/clip_vit
ENV STT_MODEL_PATH=/app/models/stt_model
ENV VQA_MODEL_PATH=/app/models/qwen_vl

# =========================
# Entrypoint
# =========================
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
 