# ── Base image (model weights pre-loaded, built via build-base-image.yml) ────
FROM ghcr.io/mindoorio-hue/wan2.2-i2v-runpod-base:latest

# ── Application code ─────────────────────────────────────────────────────────
COPY handler.py /app/handler.py

# ── Runtime ──────────────────────────────────────────────────────────────────
CMD ["python", "-u", "/app/handler.py"]
