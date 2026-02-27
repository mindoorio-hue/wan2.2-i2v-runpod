# ── Base image ──────────────────────────────────────────────────────────────
# PyTorch 2.6.0 + CUDA 12.8.1 + Ubuntu 22.04 (RunPod official)
FROM runpod/pytorch:1.0.3-cu1281-torch260-ubuntu2204

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
        "hf-transfer" \
        "sentencepiece" \
        "ftfy" \
        "opencv-python-headless" \
        "pillow" \
        "numpy"

# Use hf-transfer for fast parallel model downloads at build time
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# flash-attn for faster attention (requires CUDA dev headers from the base image)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# ── Pre-download model weights ───────────────────────────────────────────────
# Bake the model into the image so workers start instantly with no cold download.
# from_pretrained() finds it automatically via the HuggingFace local cache.
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Wan-AI/Wan2.2-I2V-A14B-Diffusers')"

# ── Application code ─────────────────────────────────────────────────────────
COPY handler.py .

# ── Runtime ──────────────────────────────────────────────────────────────────
CMD ["python", "-u", "handler.py"]
