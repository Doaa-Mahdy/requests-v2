FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y curl ffmpeg && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- PRE-CACHE HUGGINGFACE MODELS ---
# Create a script to download models to a local directory
# Copy the entire scripts directory so all internal references work
COPY scripts/ /app/scripts/

# Run the script from its new location
RUN python3 /app/scripts/download_models.py
# Copy the rest of your app
COPY . .

# Set environment to prevent HF from downloading at runtime
# These are the paths where your script downloaded them
ENV FRAUD_MODEL_PATH=/app/models/fraud_detection
ENV CLIP_MODEL_PATH=/app/models/clip_vit
ENV STT_MODEL_PATH=/app/models/stt_model
ENV VQA_MODEL_PATH=/app/models/qwen_vl

# This forces everything to be local only
ENV TRANSFORMERS_OFFLINE=1

# Start script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
ENTRYPOINT ["./entrypoint.sh"]