#!/usr/bin/env bash
###############################################################################
# client.sh — Launch Open WebUI in Docker, connected to the remote GLM-5
#              llama.cpp server
#
# Run this on your local desktop/laptop (not on the server).
# Requires: Docker installed and running
#
# Usage:  chmod +x client.sh && ./client.sh
#         Then open http://localhost:3000 in your browser
###############################################################################

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
# RunPod proxy URL for the llama.cpp server
LLAMA_SERVER_URL="https://d2p9nb0mef2kj5-8080.proxy.runpod.net/v1"

# API key for the llama.cpp server
API_KEY=""

# Local port for Open WebUI
WEBUI_PORT="3000"

# Container name
CONTAINER_NAME="open-webui"

# Open WebUI Docker image
WEBUI_IMAGE="ghcr.io/open-webui/open-webui:main"
# ─────────────────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[x]${NC} $*" >&2; }

# ─── Preflight checks ───────────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    err "Docker is not installed. Install it from https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker info &>/dev/null 2>&1; then
    err "Docker daemon is not running. Start Docker and try again."
    exit 1
fi

# ─── Handle existing container ───────────────────────────────────────────────
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        warn "Container '${CONTAINER_NAME}' is already running on port ${WEBUI_PORT}"
        log "Open http://localhost:${WEBUI_PORT} in your browser"
        exit 0
    else
        warn "Removing stopped container '${CONTAINER_NAME}'..."
        docker rm "${CONTAINER_NAME}" >/dev/null
    fi
fi

# ─── Pull latest image ──────────────────────────────────────────────────────
log "Pulling latest Open WebUI image..."
docker pull "${WEBUI_IMAGE}"

# ─── Launch Open WebUI ───────────────────────────────────────────────────────
log "Starting Open WebUI container..."

docker run -d \
    --name "${CONTAINER_NAME}" \
    --restart unless-stopped \
    -p "${WEBUI_PORT}:8080" \
    -v open-webui-data:/app/backend/data \
    -e OPENAI_API_BASE_URL="${LLAMA_SERVER_URL}" \
    -e OPENAI_API_KEY="${API_KEY}" \
    -e OLLAMA_BASE_URL="" \
    -e ENABLE_OLLAMA_API=false \
    -e WEBUI_AUTH=true \
    "${WEBUI_IMAGE}"

# ─── Wait for WebUI to start ────────────────────────────────────────────────
echo -n "Waiting for Open WebUI to start..."
MAX_WAIT=60
WAITED=0
while [[ $WAITED -lt $MAX_WAIT ]]; do
    if curl -s "http://localhost:${WEBUI_PORT}" >/dev/null 2>&1; then
        echo ""
        log "Open WebUI is ready!"
        break
    fi
    echo -n "."
    sleep 2
    WAITED=$((WAITED + 2))
done

if [[ $WAITED -ge $MAX_WAIT ]]; then
    warn "Open WebUI may still be starting. Check: docker logs ${CONTAINER_NAME}"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}=================================================================${NC}"
echo -e "${GREEN}  Open WebUI is running!${NC}"
echo -e "${GREEN}=================================================================${NC}"
echo ""
echo "  URL:            http://localhost:${WEBUI_PORT}"
echo "  Backend:        ${LLAMA_SERVER_URL}"
echo "  Model:          GLM-5 Q4_K_M (744B MoE, 4-bit)"
echo ""
echo "  First time setup:"
echo "    1. Open http://localhost:${WEBUI_PORT} in your browser"
echo "    2. Create an admin account (first signup becomes admin)"
echo "    3. Select the GLM-5 model from the model dropdown"
echo "    4. Start chatting!"
echo ""
echo "  Management:"
echo "    Stop:          docker stop ${CONTAINER_NAME}"
echo "    Start:         docker start ${CONTAINER_NAME}"
echo "    Logs:          docker logs -f ${CONTAINER_NAME}"
echo "    Remove:        docker rm -f ${CONTAINER_NAME}"
echo "    Remove data:   docker volume rm open-webui-data"
echo ""
