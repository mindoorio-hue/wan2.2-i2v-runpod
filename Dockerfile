# ── Base image ──────────────────────────────────────────────────────────────
# PyTorch 2.4 + CUDA 12.4 + Python 3.11 on Ubuntu 22.04 (RunPod official)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ──────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
        "runpod>=1.7.0" \
        "diffusers>=0.32.0" \
        "transformers>=4.46.0" \
        "accelerate>=1.0.0" \
        "huggingface_hub>=0.24.0" \
        "sentencepiece" \
        "pillow" \
        "numpy"

# flash-attn for faster attention (requires CUDA dev headers from the base image)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# ── Application code ─────────────────────────────────────────────────────────
COPY handler.py .

# ── Runtime ──────────────────────────────────────────────────────────────────
CMD ["python", "-u", "handler.py"]
