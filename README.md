# GLM-5 4-bit (Q4_K_M) — llama.cpp Server Setup

Running [GLM-5](https://huggingface.co/THUDM/GLM-5) (744B MoE, 40B active parameters) with 4-bit quantization via [llama.cpp](https://github.com/ggml-org/llama.cpp), served over an OpenAI-compatible API on port 8080.

Model weights: [unsloth/GLM-5-GGUF](https://huggingface.co/unsloth/GLM-5-GGUF)

## Hardware

- **GPUs**: 5x NVIDIA RTX PRO 6000 Blackwell Server Edition (~98 GB VRAM each, ~490 GB total)
- **CPU**: Intel Xeon 6952P (96 cores / 192 threads)
- **CUDA**: 13.0, Driver 580.126.09

## Observed Resource Usage

| GPU | VRAM Used | VRAM Total |
|-----|-----------|------------|
| 0   | 77,345 MiB | 97,887 MiB |
| 1   | 90,171 MiB | 97,887 MiB |
| 2   | 90,963 MiB | 97,887 MiB |
| 3   | 90,171 MiB | 97,887 MiB |
| 4   | 85,113 MiB | 97,887 MiB |
| **Total** | **~433 GB** | **~490 GB** |

Generation speed: ~20-37 tokens/sec (varies with prompt cache hits).

## Prerequisites

| Tool | Version used |
|------|-------------|
| git | any recent |
| cmake | >= 3.14 |
| gcc / g++ | 13.x |
| CUDA toolkit | 12.8+ |
| Python 3 | 3.12 |
| huggingface_hub | >= 1.x |

## Step 1 -- Clone and build llama.cpp with CUDA

```bash
cd /workspace
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Configure with CUDA for Blackwell (sm_100)
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=100

# Build with all available cores
cmake --build build -j$(nproc)
```

The compiled binaries are placed in `build/bin/`, including `llama-server`.

## Step 2 -- Download GLM-5 Q4_K_M GGUF

The model is hosted at [unsloth/GLM-5-GGUF](https://huggingface.co/unsloth/GLM-5-GGUF). The Q4_K_M quantization consists of 11 split GGUF files totaling ~426 GB on disk.

```bash
mkdir -p /workspace/models/GLM-5-Q4_K_M

# Set your Hugging Face token
export HF_TOKEN="hf_YOUR_TOKEN_HERE"

# Download using the huggingface_hub Python library
python3 -c "
from huggingface_hub import snapshot_download
import os

snapshot_download(
    repo_id='unsloth/GLM-5-GGUF',
    allow_patterns='Q4_K_M/*',
    local_dir='/workspace/models/GLM-5-Q4_K_M',
    token=os.environ['HF_TOKEN'],
)
print('Download complete')
"
```

After download, the GGUF split files will be in:

```
/workspace/models/GLM-5-Q4_K_M/Q4_K_M/GLM-5-Q4_K_M-00001-of-00011.gguf   (9 MB, metadata)
/workspace/models/GLM-5-Q4_K_M/Q4_K_M/GLM-5-Q4_K_M-00002-of-00011.gguf   (~47 GB)
/workspace/models/GLM-5-Q4_K_M/Q4_K_M/GLM-5-Q4_K_M-00003-of-00011.gguf   (~46 GB)
...
/workspace/models/GLM-5-Q4_K_M/Q4_K_M/GLM-5-Q4_K_M-00011-of-00011.gguf   (~14 GB)
```

## Step 3 -- Run the server

Launch `llama-server` across all 5 GPUs, bound to `0.0.0.0:8080` so it is externally accessible:

```bash
/workspace/llama.cpp/build/bin/llama-server \
    -m /workspace/models/GLM-5-Q4_K_M/Q4_K_M/GLM-5-Q4_K_M-00001-of-00011.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -ngl 999 \
    --ctx-size 8192 \
    --flash-attn on \
    --split-mode layer \
    --tensor-split 1,1,1,1,1 \
    --api-key 07bbc8c925648141d45ad9c363febf6cedc059744b5831e1
```

### Flag reference

| Flag | Purpose |
|------|---------|
| `-m` | Path to the first GGUF split file (llama.cpp auto-loads the rest) |
| `--host 0.0.0.0` | Bind to all interfaces (externally accessible) |
| `--port 8080` | Listen on port 8080 |
| `-ngl 999` | Offload all layers to GPU (model has 80 layers, 999 ensures all are offloaded) |
| `--ctx-size 8192` | Context window size in tokens (increase if VRAM headroom allows) |
| `--flash-attn on` | Enable flash attention (requires explicit `on`/`off`/`auto` value) |
| `--split-mode layer` | Split model layers across GPUs |
| `--tensor-split 1,1,1,1,1` | Distribute evenly across all 5 GPUs |
| `--api-key KEY` | Require Bearer token auth on inference endpoints (public: `/health`, `/v1/models`) |

### Important notes

- GLM-5 is a **reasoning model**. Responses include a `reasoning_content` field (chain-of-thought) and a `content` field (final answer). The thinking phase can consume significant tokens before the model produces visible output.
- The `--flash-attn` flag requires an explicit value (`on`, `off`, or `auto`). Using `--flash-attn` without a value will cause a parse error.
- llama.cpp automatically detects and loads all split GGUF files in sequence when given the first file.

## Step 4 -- Verify the server

Once the server logs `server is listening on http://0.0.0.0:8080`, test it:

```bash
# Health check (no auth required)
curl http://localhost:8080/health
# Expected: {"status":"ok"}

# Chat completion (requires API key)
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 07bbc8c925648141d45ad9c363febf6cedc059744b5831e1" \
    -d '{
        "model": "glm-5",
        "messages": [
            {"role": "user", "content": "Hello, who are you?"}
        ],
        "max_tokens": 1024
    }'
```

Requests without a valid API key receive `401 Invalid API Key`. The `/health` and `/v1/models` endpoints are public (no auth).

Note: Use a higher `max_tokens` (e.g., 1024+) because the model's internal reasoning/thinking phase consumes tokens before producing visible output in the `content` field.

## API Endpoints

The llama.cpp server exposes an OpenAI-compatible API:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Server health status |
| `POST /v1/chat/completions` | Chat completions (OpenAI-compatible) |
| `POST /v1/completions` | Text completions |
| `GET /v1/models` | List loaded models |
| `POST /tokenize` | Tokenize text |
| `POST /detokenize` | Detokenize tokens |

## External Access (RunPod)

This server runs on RunPod (pod ID: `d2p9nb0mef2kj5`). The container listens on `0.0.0.0:8080` internally, but RunPod uses a proxy for external access.

**Proxy URL (recommended):**
```
https://d2p9nb0mef2kj5-8080.proxy.runpod.net
```

Ensure port 8080 is listed in your pod's "Expose HTTP Ports" configuration in the RunPod dashboard.

### Connecting from a remote machine

**curl:**
```bash
curl https://d2p9nb0mef2kj5-8080.proxy.runpod.net/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer 07bbc8c925648141d45ad9c363febf6cedc059744b5831e1" \
    -d '{
        "model": "glm-5",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 1024
    }'
```

**Python (OpenAI client):**
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://d2p9nb0mef2kj5-8080.proxy.runpod.net/v1",
    api_key="07bbc8c925648141d45ad9c363febf6cedc059744b5831e1",
)
response = client.chat.completions.create(
    model="glm-5",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=1024,
)
print(response.choices[0].message.content)
```

**SSH tunnel (alternative):**
```bash
ssh -L 8080:localhost:8080 root@157.254.50.91 -p 11249
# Then connect via http://localhost:8080
```

## Open WebUI (Chat Interface)

[Open WebUI](https://github.com/open-webui/open-webui) provides a ChatGPT-like web interface that connects to the llama.cpp server. Run it on your local desktop via Docker.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed and running on your local machine

### Quick start

```bash
chmod +x client.sh && ./client.sh
```

Then open http://localhost:3000 in your browser. On first visit, create an admin account (first signup becomes admin), then select the GLM-5 model from the dropdown.

### Manual Docker run

```bash
docker run -d \
    --name open-webui \
    --restart unless-stopped \
    -p 3000:8080 \
    -v open-webui-data:/app/backend/data \
    -e OPENAI_API_BASE_URL="https://d2p9nb0mef2kj5-8080.proxy.runpod.net/v1" \
    -e OPENAI_API_KEY="07bbc8c925648141d45ad9c363febf6cedc059744b5831e1" \
    -e OLLAMA_BASE_URL="" \
    -e ENABLE_OLLAMA_API=false \
    ghcr.io/open-webui/open-webui:main
```

### Environment variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `OPENAI_API_BASE_URL` | `https://d2p9nb0mef2kj5-8080.proxy.runpod.net/v1` | Points to the llama.cpp server |
| `OPENAI_API_KEY` | `07bbc8c925648141d45ad9c363febf6cedc059744b5831e1` | API key for authentication |
| `ENABLE_OLLAMA_API` | `false` | Disable Ollama (not used) |

### Management

```bash
docker stop open-webui       # stop
docker start open-webui      # start
docker logs -f open-webui    # view logs
docker rm -f open-webui      # remove container
docker volume rm open-webui-data  # remove all data
```

## Troubleshooting

- **Out of VRAM**: Reduce `--ctx-size` or use a smaller quantization (e.g., Q3_K_M or the Unsloth dynamic 2-bit `UD-IQ2_XXS`).
- **Slow inference**: Ensure all layers are GPU-offloaded (`-ngl 999`). Check `nvidia-smi` to verify GPU utilization.
- **Split files not found**: llama.cpp automatically loads split files if they follow the naming convention `*-NNNNN-of-NNNNN.gguf` and are in the same directory.
- **Empty `content` in response**: GLM-5 is a reasoning model. Increase `max_tokens` to allow the model to finish its thinking phase and produce a final answer. With 512 tokens you may only see `reasoning_content`.
- **`--flash-attn` parse error**: This flag now requires an explicit value: `--flash-attn on`, not just `--flash-attn`.

## Quick Start (Fresh Server)

To set up everything from scratch on a new Ubuntu server with NVIDIA GPUs:

```bash
chmod +x install.sh && sudo ./install.sh
```

This script handles all 7 steps automatically: system packages, NVIDIA/CUDA drivers, Node.js + Claude Code, Python + HuggingFace, llama.cpp build, model download (~426 GB), and server launch. It is idempotent -- safe to re-run (skips already-completed steps).

## File Layout

```
/workspace/
  install.sh                  # server setup script (run on fresh GPU server)
  client.sh                   # Open WebUI launcher (run on local desktop)
  README.md                   # this file
  llama-server.log            # server stdout/stderr (created at runtime)
  llama-server.pid            # server PID file (created at runtime)
  llama.cpp/                  # llama.cpp source and build
    build/bin/llama-server    # compiled server binary
  models/
    GLM-5-Q4_K_M/
      Q4_K_M/                 # 11 split GGUF files (~426 GB total)
```
