#!/bin/bash
set -e

echo "Starting Ollama..."

ollama serve &
OLLAMA_PID=$!

echo "Waiting for Ollama..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
  sleep 2
done

echo "Ollama ready."

# safety check (no re-download if exists)
if ! ollama list | grep -q "qwen2.5:7b"; then
  echo "Pulling model..."
  ollama pull qwen2.5:7b
fi

echo "Starting app..."
exec python3 main.py