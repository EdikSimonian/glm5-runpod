#!/usr/bin/env bash
###############################################################################
# install.sh — Full setup: system deps, CUDA, Claude Code, HuggingFace CLI,
#              llama.cpp (CUDA), and GLM-5 Q4_K_M download + server launch
#
# Target: Ubuntu 22.04 / 24.04, fresh server with NVIDIA GPUs
# Tested: RunPod with NVIDIA RTX PRO 6000 Blackwell (5x ~98 GB VRAM)
# Usage:  chmod +x install.sh && sudo ./install.sh
###############################################################################

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
WORKSPACE="/workspace"
LLAMA_CPP_DIR="${WORKSPACE}/llama.cpp"
MODEL_DIR="${WORKSPACE}/models/GLM-5-Q4_K_M"
MODEL_REPO="unsloth/GLM-5-GGUF"
MODEL_QUANT="Q4_K_M"
SERVER_HOST="0.0.0.0"
SERVER_PORT="8080"
CTX_SIZE="8192"

HF_TOKEN=""
CLAUDE_ACCESS_TOKEN=""
CLAUDE_REFRESH_TOKEN=""
# ─────────────────────────────────────────────────────────────────────────────

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()  { echo -e "${GREEN}[+]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }
err()  { echo -e "${RED}[x]${NC} $*" >&2; }

step() {
    echo ""
    echo -e "${GREEN}=================================================================${NC}"
    echo -e "${GREEN}  $*${NC}"
    echo -e "${GREEN}=================================================================${NC}"
}

# ─── Check root ──────────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    err "This script must be run as root (sudo ./install.sh)"
    exit 1
fi

# ─── Step 1: System packages ────────────────────────────────────────────────
step "Step 1/7: Installing system packages"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    python3 \
    python3-pip \
    python3-venv \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    gnupg \
    lsb-release \
    lsof \
    psmisc \
    jq

log "System packages installed"

# ─── Step 2: NVIDIA driver + CUDA toolkit ────────────────────────────────────
step "Step 2/7: Setting up NVIDIA drivers and CUDA toolkit"

if command -v nvidia-smi &>/dev/null; then
    log "NVIDIA driver already installed: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
    warn "No NVIDIA driver detected. Installing..."
    DISTRO="ubuntu$(lsb_release -rs | tr -d '.')"
    ARCH="$(uname -m)"
    wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
    dpkg -i /tmp/cuda-keyring.deb
    apt-get update -qq
    apt-get install -y -qq nvidia-driver-580
    log "NVIDIA driver installed. A REBOOT may be required before continuing."
fi

if command -v nvcc &>/dev/null; then
    log "CUDA toolkit already installed: $(nvcc --version | grep release)"
else
    warn "CUDA toolkit not found. Installing CUDA 12.8..."
    DISTRO="ubuntu$(lsb_release -rs | tr -d '.')"
    ARCH="$(uname -m)"
    if [[ ! -f /tmp/cuda-keyring.deb ]]; then
        wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb
        dpkg -i /tmp/cuda-keyring.deb
        apt-get update -qq
    fi
    apt-get install -y -qq cuda-toolkit-12-8
    log "CUDA toolkit installed"
fi

# Ensure CUDA is on PATH for the rest of this script (idempotent)
if [[ ! -f /etc/profile.d/cuda.sh ]]; then
    cat > /etc/profile.d/cuda.sh << 'CUDA_EOF'
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
CUDA_EOF
fi
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# ─── Step 3: Node.js + Claude Code ──────────────────────────────────────────
step "Step 3/7: Installing Node.js and Claude Code"

if command -v node &>/dev/null && [[ "$(node --version | cut -d. -f1 | tr -d 'v')" -ge 18 ]]; then
    log "Node.js already installed: $(node --version)"
else
    warn "Installing Node.js 22 LTS..."
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt-get install -y -qq nodejs
    log "Node.js installed: $(node --version)"
fi

if command -v claude &>/dev/null; then
    log "Claude Code already installed"
else
    npm install -g @anthropic-ai/claude-code
    log "Claude Code installed"
fi

# Configure Claude Code credentials
mkdir -p /root/.claude
cat > /root/.claude/.credentials.json << CRED_EOF
{
  "claudeAiOauth": {
    "accessToken": "${CLAUDE_ACCESS_TOKEN}",
    "refreshToken": "${CLAUDE_REFRESH_TOKEN}",
    "scopes": [
      "user:inference",
      "user:mcp_servers",
      "user:profile",
      "user:sessions:claude_code"
    ],
    "subscriptionType": "pro",
    "rateLimitTier": "default_claude_ai"
  }
}
CRED_EOF
chmod 600 /root/.claude/.credentials.json

cat > /root/.claude/settings.json << 'SETTINGS_EOF'
{
  "effortLevel": "high"
}
SETTINGS_EOF

log "Claude Code credentials and settings configured"

# ─── Step 4: Python + HuggingFace CLI ───────────────────────────────────────
step "Step 4/7: Setting up Python and HuggingFace"

# --break-system-packages is needed on Ubuntu 24.04+ (PEP 668)
PIP_BREAK=""
if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)" 2>/dev/null; then
    PIP_BREAK="--break-system-packages"
fi
pip3 install --quiet ${PIP_BREAK} huggingface_hub openai
log "Python packages installed (huggingface_hub, openai)"

# Store HF token for future use
mkdir -p /root/.cache/huggingface
printf '%s' "${HF_TOKEN}" > /root/.cache/huggingface/token
chmod 600 /root/.cache/huggingface/token

log "HuggingFace token configured"

# ─── Step 5: Clone and build llama.cpp ───────────────────────────────────────
step "Step 5/7: Building llama.cpp with CUDA"

mkdir -p "${WORKSPACE}"

if [[ -d "${LLAMA_CPP_DIR}/.git" ]]; then
    warn "llama.cpp directory exists, pulling latest..."
    cd "${LLAMA_CPP_DIR}"
    git pull --ff-only || true
else
    rm -rf "${LLAMA_CPP_DIR}"
    cd "${WORKSPACE}"
    git clone https://github.com/ggml-org/llama.cpp.git
fi

cd "${LLAMA_CPP_DIR}"

# Detect GPU architecture for cmake
# Blackwell = 120, Hopper = 90, Ada = 89, Ampere = 80/86
GPU_ARCH=""
if nvidia-smi &>/dev/null; then
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
    if [[ -n "$COMPUTE_CAP" ]]; then
        GPU_ARCH="${COMPUTE_CAP}"
        log "Detected GPU compute capability: ${COMPUTE_CAP}"
    fi
fi

CMAKE_CUDA_ARGS=""
if [[ -n "$GPU_ARCH" ]]; then
    CMAKE_CUDA_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${GPU_ARCH}"
fi

cmake -B build -DGGML_CUDA=ON ${CMAKE_CUDA_ARGS}
cmake --build build -j"$(nproc)"

log "llama.cpp built successfully"
log "Server binary: ${LLAMA_CPP_DIR}/build/bin/llama-server"

# ─── Step 6: Download GLM-5 Q4_K_M ──────────────────────────────────────────
step "Step 6/7: Downloading GLM-5 Q4_K_M GGUF (~426 GB)"

mkdir -p "${MODEL_DIR}"

FIRST_SHARD="${MODEL_DIR}/${MODEL_QUANT}/GLM-5-${MODEL_QUANT}-00001-of-00011.gguf"

if [[ -f "${FIRST_SHARD}" ]]; then
    SHARD_COUNT=$(ls "${MODEL_DIR}/${MODEL_QUANT}"/GLM-5-${MODEL_QUANT}-*.gguf 2>/dev/null | wc -l)
    if [[ "$SHARD_COUNT" -eq 11 ]]; then
        log "All 11 GGUF shards already present, skipping download"
    else
        warn "Partial download detected (${SHARD_COUNT}/11 shards). Resuming..."
        HF_TOKEN="${HF_TOKEN}" python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='${MODEL_REPO}',
    allow_patterns='${MODEL_QUANT}/*',
    local_dir='${MODEL_DIR}',
    token=os.environ['HF_TOKEN'],
)
print('Download complete')
"
    fi
else
    warn "This will download ~426 GB. Ensure you have enough disk space."
    df -h "${WORKSPACE}" | tail -1
    echo ""

    HF_TOKEN="${HF_TOKEN}" python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='${MODEL_REPO}',
    allow_patterns='${MODEL_QUANT}/*',
    local_dir='${MODEL_DIR}',
    token=os.environ['HF_TOKEN'],
)
print('Download complete')
"
fi

log "GLM-5 Q4_K_M model ready at ${MODEL_DIR}/${MODEL_QUANT}/"

# ─── Step 7: Launch the server ───────────────────────────────────────────────
step "Step 7/7: Launching llama.cpp server"

# Detect number of GPUs for tensor split
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
TENSOR_SPLIT=$(python3 -c "print(','.join(['1']*${NUM_GPUS}))")
log "Detected ${NUM_GPUS} GPUs, tensor split: ${TENSOR_SPLIT}"

# Kill any existing server on the same port
if lsof -i ":${SERVER_PORT}" &>/dev/null; then
    warn "Port ${SERVER_PORT} is in use, killing existing process..."
    fuser -k "${SERVER_PORT}/tcp" 2>/dev/null || true
    sleep 2
fi

# Launch server in background
nohup "${LLAMA_CPP_DIR}/build/bin/llama-server" \
    -m "${MODEL_DIR}/${MODEL_QUANT}/GLM-5-${MODEL_QUANT}-00001-of-00011.gguf" \
    --host "${SERVER_HOST}" \
    --port "${SERVER_PORT}" \
    -ngl 999 \
    --ctx-size "${CTX_SIZE}" \
    --flash-attn on \
    --split-mode layer \
    --tensor-split "${TENSOR_SPLIT}" \
    > "${WORKSPACE}/llama-server.log" 2>&1 &

SERVER_PID=$!
echo "${SERVER_PID}" > "${WORKSPACE}/llama-server.pid"
log "Server starting in background (PID: ${SERVER_PID})"
log "Log file: ${WORKSPACE}/llama-server.log"

# Wait for server to become ready
echo -n "Waiting for server to load model and start listening..."
MAX_WAIT=600
WAITED=0
while [[ $WAITED -lt $MAX_WAIT ]]; do
    if curl -s "http://localhost:${SERVER_PORT}/health" 2>/dev/null | grep -q '"ok"'; then
        echo ""
        log "Server is ready!"
        break
    fi
    # Check if process is still alive
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo ""
        err "Server process died. Check ${WORKSPACE}/llama-server.log for details:"
        tail -20 "${WORKSPACE}/llama-server.log"
        exit 1
    fi
    echo -n "."
    sleep 5
    WAITED=$((WAITED + 5))
done

if [[ $WAITED -ge $MAX_WAIT ]]; then
    err "Server did not start within ${MAX_WAIT}s. Check ${WORKSPACE}/llama-server.log"
    exit 1
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
INTERNAL_IP=$(hostname -I | awk '{print $1}')

echo ""
echo -e "${GREEN}=================================================================${NC}"
echo -e "${GREEN}  Setup Complete!${NC}"
echo -e "${GREEN}=================================================================${NC}"
echo ""
echo "  Model:        GLM-5 Q4_K_M (744B MoE, 4-bit)"
echo "  Internal:     http://${INTERNAL_IP}:${SERVER_PORT}"
echo "  PID file:     ${WORKSPACE}/llama-server.pid"
echo "  Log file:     ${WORKSPACE}/llama-server.log"
echo ""

# Detect RunPod environment and show proxy URL
if [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    PROXY_URL="https://${RUNPOD_POD_ID}-${SERVER_PORT}.proxy.runpod.net"
    echo "  RunPod proxy: ${PROXY_URL}"
    echo ""
    echo "  Health check:"
    echo "    curl ${PROXY_URL}/health"
    echo ""
    echo "  Chat completion:"
    echo "    curl ${PROXY_URL}/v1/chat/completions \\"
    echo "      -H 'Content-Type: application/json' \\"
    echo "      -d '{\"model\":\"glm-5\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":1024}'"
    echo ""
    echo "  Python (OpenAI client):"
    echo "    from openai import OpenAI"
    echo "    client = OpenAI(base_url=\"${PROXY_URL}/v1\", api_key=\"not-needed\")"
    echo ""
    if [[ -n "${RUNPOD_PUBLIC_IP:-}" && -n "${RUNPOD_TCP_PORT_22:-}" ]]; then
        echo "  SSH tunnel (alternative):"
        echo "    ssh -L 8080:localhost:8080 root@${RUNPOD_PUBLIC_IP} -p ${RUNPOD_TCP_PORT_22}"
        echo "    Then connect via http://localhost:8080"
        echo ""
    fi
else
    PUBLIC_IP="${RUNPOD_PUBLIC_IP:-${INTERNAL_IP}}"
    echo "  Health check:"
    echo "    curl http://${PUBLIC_IP}:${SERVER_PORT}/health"
    echo ""
    echo "  Chat completion:"
    echo "    curl http://${PUBLIC_IP}:${SERVER_PORT}/v1/chat/completions \\"
    echo "      -H 'Content-Type: application/json' \\"
    echo "      -d '{\"model\":\"glm-5\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"max_tokens\":1024}'"
    echo ""
fi

echo "  Stop server:  kill \$(cat ${WORKSPACE}/llama-server.pid)"
echo "  View logs:    tail -f ${WORKSPACE}/llama-server.log"
echo "  Claude Code:  claude"
echo ""
