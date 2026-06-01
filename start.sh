#!/bin/bash
set -e

echo "=================================================="
echo "          STARTING TELEMETRY AI STACK             "
echo "=================================================="

# 1. ENSURE DOCKER NETWORK EXISTS
if ! docker network inspect telem-net >/dev/null 2>&1; then
    echo "Creating isolated Docker network: telem-net..."
    docker network create telem-net
fi

# 2. CLEAN UP OLD INSTANCES (Prevents conflicts)
echo "Cleaning up any stale telemetry containers..."
docker-compose -f docker-compose.edge.yml down
docker stop telemetry-app ollama-server mosquitto-broker hardware-publisher 2>/dev/null || true
docker rm telemetry-app ollama-server mosquitto-broker hardware-publisher 2>/dev/null || true

# 3. DETECT GPU ARCHITECTURE
if command -v nvidia-smi &> /dev/null; then
    export GPU_TYPE="NVIDIA"
    echo "SUCCESS: NVIDIA GPU detected."
elif lspci | grep -i "vga" | grep -i "AMD" &> /dev/null; then
    export GPU_TYPE="AMD"
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    echo "SUCCESS: AMD GPU detected. Target: gfx1030"
else
    export GPU_TYPE="CPU"
    echo "WARNING: No dedicated GPU found."
fi

# 4. LAUNCH VIA DOCKER COMPOSE
echo "Launching Edge Infrastructure..."
export APP_MODE="edge"
# This single command spins up FastAPI, Mosquitto, Ollama, AND the Hardware Publisher!
docker-compose -f docker-compose.edge.yml up --build -d

# 5. SYNC CHAT COMPLETION MODELS
echo "Waiting for Ollama API handshake to initialize..."
until docker exec ollama-server ollama list >/dev/null 2>&1; do
    sleep 2
done

echo "Verifying local LLM assets (Gemma 3)..."
docker exec -it ollama-server ollama pull gemma3:4b



echo "=================================================="
echo "  System Online!                                  "
echo "  View Dashboard: http://localhost:8000           "
echo "=================================================="