# 1. Use the official Ollama image
FROM ollama/ollama:latest

# 2. Install essential build tools and system libraries
# This is the fix for "exit code 1"
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

# 3. Upgrade pip and install dependencies
# We use pip3 to ensure we are using the system Python
COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# 4. Copy your app code and pre-cache models
COPY app/scripts/ /app/scripts/
RUN python3 /app/scripts/download_models.py
COPY . .

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