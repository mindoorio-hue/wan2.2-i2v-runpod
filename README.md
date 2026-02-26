# Wan2.2 Image-to-Video — RunPod Serverless

Generate cinematic videos from a single image using **[Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers)** — a 27B Mixture-of-Experts model (14B active parameters) that supports 480P and 720P output.

---

## Hardware Requirements

| Resolution | Recommended GPU       | Min VRAM |
|------------|-----------------------|----------|
| 480P       | A100 80GB / H100 80GB | ~40 GB   |
| 720P       | A100 80GB / H100 80GB | ~80 GB   |
| 480P (CPU offload) | RTX 4090 (24 GB) | ~24 GB |

---

## API Input

Send a POST request to your endpoint with a JSON body:

```json
{
  "input": {
    "image":               "<base64-encoded image>",
    "prompt":              "A cat gently swishing its tail on a wooden floor.",
    "negative_prompt":     "low quality, blurry, watermark",
    "resolution":          "480p",
    "num_frames":          81,
    "guidance_scale":      3.5,
    "num_inference_steps": 40,
    "fps":                 16,
    "seed":                42
  }
}
```

### Input Fields

| Field                  | Type    | Default      | Description                                                  |
|------------------------|---------|--------------|--------------------------------------------------------------|
| `image`                | string  | **required** | Base64-encoded input image (JPEG or PNG). Data-URI prefix optional. |
| `prompt`               | string  | `""`         | Text description of the desired motion/scene.                |
| `negative_prompt`      | string  | (see below)  | Concepts to avoid in the generated video.                    |
| `resolution`           | string  | `"480p"`     | Output resolution: `"480p"` or `"720p"`.                     |
| `num_frames`           | integer | `81`         | Number of video frames (~5 s at 16 fps).                     |
| `guidance_scale`       | float   | `3.5`        | Classifier-free guidance strength.                           |
| `num_inference_steps`  | integer | `40`         | Denoising steps (more = higher quality, slower).             |
| `fps`                  | integer | `16`         | Frames per second of the exported MP4.                       |
| `seed`                 | integer | random       | RNG seed for reproducible results.                           |

**Default negative prompt:**
```
low quality, blurry, jittery, distorted, static, overexposed,
watermark, text, logo, artifacts, worst quality, bad anatomy
```

---

## API Output

```json
{
  "video_base64": "<base64-encoded MP4>",
  "width":        832,
  "height":       480,
  "num_frames":   81,
  "fps":          16,
  "resolution":   "480p"
}
```

Decode the `video_base64` field to obtain the raw MP4 file:

```python
import base64

data = response["output"]["video_base64"]
with open("output.mp4", "wb") as f:
    f.write(base64.b64decode(data))
```

---

## Quick-Start — Python Client

```python
import runpod, base64, time

runpod.api_key = "YOUR_RUNPOD_API_KEY"

with open("my_image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")

run = endpoint.run({
    "image":               image_b64,
    "prompt":              "Ocean waves gently rolling onto a sandy shore at sunset.",
    "resolution":          "480p",
    "num_frames":          81,
    "num_inference_steps": 40,
    "guidance_scale":      3.5,
    "fps":                 16,
    "seed":                42,
})

output = run.output(timeout=600)

with open("output.mp4", "wb") as f:
    f.write(base64.b64decode(output["video_base64"]))

print("Saved output.mp4")
```

---

## Environment Variables

| Variable              | Default                               | Description                                      |
|-----------------------|---------------------------------------|--------------------------------------------------|
| `MODEL_ID`            | `Wan-AI/Wan2.2-I2V-A14B-Diffusers`   | HuggingFace model ID (locked; do not change).    |
| `RESOLUTION`          | `480p`                                | Default resolution when not specified per-request.|
| `ENABLE_CPU_OFFLOAD`  | `false`                               | Set `true` to enable CPU offload for lower VRAM. |
| `HF_TOKEN`            | _(empty)_                             | HuggingFace token for private/gated models.      |

---

## Repository Structure

```
.
├── .runpod/
│   ├── hub.json        # RunPod Hub metadata & deployment config
│   └── tests.json      # Automated test cases
├── handler.py          # RunPod serverless handler
├── Dockerfile          # Container build definition
├── icon.png            # Hub listing icon
└── README.md           # This file
```

---

## Model License

The Wan2.2 model is released under the [Wan Community License](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers/blob/main/LICENSE). Please review the license terms before commercial use.

---

[![RunPod Hub](https://api.runpod.io/badge/YOUR_USERNAME/YOUR_REPO)](https://runpod.io/hub)
