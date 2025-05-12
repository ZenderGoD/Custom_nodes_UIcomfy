import requests
import json
import base64
import os
import io
import time
import numpy as np
import torch
from PIL import Image

# Helper function to get API key (from input or environment)
def get_api_key(api_key_input):
    if api_key_input and api_key_input.strip().lower() not in ["your_aimlapi_key", ""]:
        return api_key_input.strip()
    return os.environ.get('AIMLAPI_KEY')

# Create a fallback red image tensor (for debugging)
def fallback_image_tensor():
    fallback_np = np.ones((64, 64, 3), dtype=np.float32)
    fallback_np[:, :, 1:] = 0  # Red only
    return torch.from_numpy(fallback_np).unsqueeze(0)

# ComfyUI Node
class AIMLAPIFluxGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Epic cinematic shot of a dragon flying over a futuristic city"}),
                "api_key": ("STRING", {"multiline": False, "default": "YOUR_AIMLAPI_KEY"}),
                "aspect_ratio": (["16:9", "1:1", "9:16", "4:3", "3:4", "21:9", "9:21"], {"default": "16:9"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "model_id": ("STRING", {"default": "flux-pro/v1.1-ultra"}),
                "output_format": (["jpeg", "png"], {"default": "jpeg"}),
                "safety_tolerance": (["1", "2", "3", "4", "5", "6"], {"default": "1"}),
                "enable_safety_checker": (["enable", "disable"], {"default": "enable"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "API/AIMLAPI"

    def generate_image(self, prompt, api_key, aspect_ratio, seed,
                       model_id="flux-pro/v1.1-ultra",
                       output_format="jpeg",
                       safety_tolerance="1",
                       enable_safety_checker="enable"):

        image_tensor = fallback_image_tensor()  # Default red fallback
        key = get_api_key(api_key)
        if not key:
            print("❌ API key missing.")
            return (image_tensor,)

        if not prompt.strip():
            print("⚠️ Prompt is empty.")
            return (image_tensor,)

        url = "https://api.aimlapi.com/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model_id,
            "prompt": prompt,
            "num_images": 1,
            "output_format": output_format,
            "aspect_ratio": aspect_ratio,
            "safety_tolerance": safety_tolerance,
            "enable_safety_checker": enable_safety_checker == "enable",
            "raw": False,
            "seed": seed
        }

        print(f"➡️ Sending request to AIMLAPI ({model_id}) with seed {seed}...")
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=300)
            response.raise_for_status()
            data = response.json()
            print("✅ Response received.")

            image_b64 = None
            if "data" in data and isinstance(data["data"], list) and data["data"]:
                image_b64 = data["data"][0].get("b64_json") or data["data"][0].get("image_data") or data["data"][0].get("image")
            elif "images" in data and isinstance(data["images"], list) and data["images"]:
                image_b64 = data["images"][0].get("b64_json") or data["images"][0].get("image_data") or data["images"][0].get("image")

            if not image_b64:
                print("❌ No base64 image data in response.")
                return (image_tensor,)

            image_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(img_np).unsqueeze(0)

            print("✅ Image tensor created successfully.")
            print(f"Tensor shape: {image_tensor.shape}")
            print(f"Tensor min/max: {image_tensor.min().item()} / {image_tensor.max().item()}")

            return (image_tensor,)

        except Exception as e:
            print(f"❌ Error during image generation: {e}")
            import traceback
            traceback.print_exc()
            return (image_tensor,)

# Register Node
NODE_CLASS_MAPPINGS = {
    "AIMLAPIFluxGenerator2": AIMLAPIFluxGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIMLAPIFluxGenerator2": "Image Generator (Flux via AIMLAPI)"
}