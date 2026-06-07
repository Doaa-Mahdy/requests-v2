# 1. Use the official Ollama image as the base
FROM ollama/ollama:latest

# 2. Install Python and other necessary system tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your app code
COPY . .

# 5. Pre-cache your non-Ollama models (Fraud, CLIP, etc.)
# Ensure your download_models.py script saves them to /app/models/
RUN python3 app/scripts/download_models.py

# 6. Set environment variables
ENV TRANSFORMERS_OFFLINE=1
ENV FRAUD_MODEL_PATH=/app/models/fraud_detection
ENV CLIP_MODEL_PATH=/app/models/clip_vit
ENV STT_MODEL_PATH=/app/models/stt_model
ENV VQA_MODEL_PATH=/app/models/qwen_vl

# 7. Use the entrypoint
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# The Ollama image already defines an ENTRYPOINT, 
# but we override it to run our own logic.
ENTRYPOINT ["./entrypoint.sh"]