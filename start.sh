#!/bin/bash

echo "Starting Telemetry AI Stack..."

# Detect GPU Vendor
if command -v nvidia-smi &> /dev/null; then
    GPU_TYPE="NVIDIA"
    echo "NVIDIA GPU detected. Using CUDA configuration."
elif lspci | grep -i "vga" | grep -i "AMD" &> /dev/null; then
    GPU_TYPE="AMD"
    echo "AMD GPU detected. Using ROCm configuration."
else
    GPU_TYPE="CPU"
    echo "No supported GPU detected. Falling back to CPU (Slow)."
fi

# Stop existing container if it exists
docker stop ollama-server 2>/dev/null && docker rm ollama-server 2>/dev/null

#Start Ollama in the background
# Launch based on detection
if [ "$GPU_TYPE" == "NVIDIA" ]; then
    docker run -d \
      -v $(pwd)/ollama_data:/root/.ollama \
      --gpus all \
      --network fyp-net \
      --name ollama-server \
      ollama/ollama:latest

elif [ "$GPU_TYPE" == "AMD" ]; then
    echo "Applying GFX override for $GPU_TYPE (Target: gfx1030)..."
    docker run -d \
      -v $(pwd)/ollama_data:/root/.ollama \
      --device /dev/kfd \
      --device /dev/dri \
      --shm-size 8G \
      --network fyp-net \
      --name ollama-server \
      -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
      ollama/ollama:rocm
else
    # CPU Fallback
    docker run -d \
      -v $(pwd)/ollama_data:/root/.ollama \
      --network fyp-net \
      --name ollama-server \
      ollama/ollama:latest
fi

# Ensures LLM model is downloaded
echo "Checking Gemma 3 model..."
docker exec -it ollama-server ollama pull gemma3:4b

# Start the Telemetry App
echo "Launching FastAPI Application..."
docker run -it --rm \
  --network fyp-net \
  --env-file .env \
  --name telemetry-app \
  -p 8000:8000 \
  telemetry-app
