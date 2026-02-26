"""
RunPod Serverless Handler — Wan2.2 Image-to-Video (I2V)
Model: Wan-AI/Wan2.2-I2V-A14B-Diffusers
"""

import base64
import os
import tempfile
from io import BytesIO

import numpy as np
import runpod
import torch
from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration (from environment variables)
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "Wan-AI/Wan2.2-I2V-A14B-Diffusers")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
ENABLE_CPU_OFFLOAD = os.environ.get("ENABLE_CPU_OFFLOAD", "false").lower() == "true"
DEFAULT_RESOLUTION = os.environ.get("RESOLUTION", "480p")

DTYPE = torch.bfloat16
DEVICE = "cuda"

DEFAULT_NEGATIVE_PROMPT = (
    "low quality, blurry, jittery, distorted, static, overexposed, "
    "watermark, text, logo, artifacts, worst quality, bad anatomy"
)

# ---------------------------------------------------------------------------
# Model loading — runs once at worker startup
# ---------------------------------------------------------------------------
print(f"[wan2.2-i2v] Loading pipeline from: {MODEL_ID}")

pipe = WanImageToVideoPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    token=HF_TOKEN,
)

if ENABLE_CPU_OFFLOAD:
    print("[wan2.2-i2v] CPU offload enabled.")
    pipe.enable_model_cpu_offload()
else:
    pipe.to(DEVICE)

print("[wan2.2-i2v] Pipeline ready.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def decode_image(image_input: str) -> Image.Image:
    """Accept a base64-encoded image (with or without data-URI prefix)."""
    if "," in image_input:
        # Strip data URI header: data:image/png;base64,<data>
        image_input = image_input.split(",", 1)[1]
    image_data = base64.b64decode(image_input)
    return Image.open(BytesIO(image_data)).convert("RGB")


def calculate_dimensions(image: Image.Image, resolution: str) -> tuple[int, int]:
    """
    Compute output (width, height) that preserves aspect ratio and is
    compatible with the VAE / patch-size requirements.
    """
    max_area = 720 * 1280 if resolution == "720p" else 480 * 832
    mod_value = (
        pipe.vae_scale_factor_spatial
        * pipe.transformer.config.patch_size[1]
    )
    aspect_ratio = image.height / image.width
    height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
    width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
    return int(width), int(height)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    job_input = job["input"]

    # --- Required ---
    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "Missing required field: 'image' (base64-encoded image string)."}

    # --- Optional with defaults ---
    prompt = job_input.get("prompt", "")
    negative_prompt = job_input.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    resolution = job_input.get("resolution", DEFAULT_RESOLUTION)
    num_frames = int(job_input.get("num_frames", 81))
    guidance_scale = float(job_input.get("guidance_scale", 3.5))
    num_inference_steps = int(job_input.get("num_inference_steps", 40))
    fps = int(job_input.get("fps", 16))
    seed = job_input.get("seed", None)

    if resolution not in ("480p", "720p"):
        return {"error": f"Invalid resolution '{resolution}'. Must be '480p' or '720p'."}

    if not (1 <= num_frames <= 200):
        return {"error": "num_frames must be between 1 and 200."}

    # --- Decode image ---
    runpod.serverless.progress_update(job, "Decoding input image...")
    try:
        image = decode_image(image_b64)
    except Exception as exc:
        return {"error": f"Failed to decode image: {exc}"}

    # --- Compute dimensions ---
    width, height = calculate_dimensions(image, resolution)
    image = image.resize((width, height))

    # --- Generate ---
    runpod.serverless.progress_update(
        job,
        f"Generating {resolution} video ({width}x{height}, "
        f"{num_frames} frames, {num_inference_steps} steps)...",
    )

    generator = torch.Generator(device=DEVICE)
    if seed is not None:
        generator.manual_seed(int(seed))

    try:
        result = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        frames = result.frames[0]
    except Exception as exc:
        return {"error": f"Video generation failed: {exc}"}

    # --- Encode output ---
    runpod.serverless.progress_update(job, "Encoding output video...")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        export_to_video(frames, tmp_path, fps=fps)
        with open(tmp_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("utf-8")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return {
        "video_base64": video_b64,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
        "resolution": resolution,
    }


runpod.serverless.start({"handler": handler})
