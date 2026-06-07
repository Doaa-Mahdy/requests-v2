#!/bin/bash

# 1. Start Ollama in the background
# We must use 'ollama serve' as the base image expects it
ollama serve & 

# 2. Wait for Ollama to be ready (health check)
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags > /dev/null; do
  sleep 2
done
echo "Ollama is up!"

# 3. Pull the required model
# Using a local volume for /root/.ollama in RunPod keeps this fast
echo "Ensuring model qwen2.5:7b is pulled..."
ollama pull qwen2.5:7b

# 4. Start your application
echo "Starting Application..."
exec python3 main.py