#!/bin/bash

echo "Starting Ollama..."
ollama serve &

echo "Waiting for Ollama..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
  sleep 2
done

echo "Pulling model..."
ollama pull qwen2.5:7b

# echo "Downloading HF models (runtime safe)..."
# python3 app/scripts/download_models.py

echo "Starting app..."
exec python3 main.py