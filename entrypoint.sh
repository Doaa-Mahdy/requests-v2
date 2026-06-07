#!/bin/bash
# 1. Start Ollama in the background
ollama serve & 
PID=$!

# 2. Wait for Ollama to be ready (health check)
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags; do
  sleep 2
done

# 3. Ensure the model exists (or pulls it if it's the first time on the volume)
# If using a Network Volume, this command will be near-instant after the first run
ollama pull qwen2.5:7b

# 4. Start your actual application
exec python3 main.py