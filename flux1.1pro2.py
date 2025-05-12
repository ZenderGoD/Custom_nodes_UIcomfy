import requests
import json
import base64
import os
import io
import time
import numpy as np
import torch
from PIL import Image

def get_api_key(api_key_input):
    if api_key_input and api_key_input.strip() and api_key_input.strip().lower() not in ["your_aimlapi_key", ""]:
         return api_key_input.strip()
    env_key = os.environ.get('AIMLAPI_KEY')
    if env_key:
        return env_key
    return None

class AIMLAPIFluxGenerator:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        aspect_ratios = ["16:9", "1:1", "9:16", "4:3", "3:4", "21:9", "9:21"]
        output_formats = ["jpeg", "png"]
        safety_tolerances = ["1", "2", "3", "4", "5", "6"]
        boolean_options = ["enable", "disable"]

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Epic cinematic shot of a dragon flying over a futuristic city, dramatic lighting, hyperrealistic, 8k"}),
                "api_key": ("STRING", {"multiline": False, "default": "YOUR_AIMLAPI_KEY"}),
                "aspect_ratio": (aspect_ratios, {"default": "16:9"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "model_id": ("STRING", {"multiline": False, "default": "flux-pro/v1.1-ultra"}),
                "output_format": (output_formats, {"default": "jpeg"}),
                "safety_tolerance": (safety_tolerances, {"default": "1"}),
                "enable_safety_checker": (boolean_options, {"default": "enable"}),
                "input_image": ("IMAGE",),  # Optional input image for img2img
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "API/AIMLAPI"

    def generate_image(self, prompt, api_key, aspect_ratio, seed,
                       model_id="flux-pro/v1.1-ultra", output_format="jpeg",
                       safety_tolerance="1", enable_safety_checker="enable",
                       input_image=None):
        effective_api_key = get_api_key(api_key)
        if not effective_api_key:
            print("\033[91mERROR: AIMLAPI Key not found. Please provide it in the node input or set the AIMLAPI_KEY environment variable.\033[0m")
            return (torch.zeros(1, 3, 64, 64, dtype=torch.float32),)

        if not prompt or not prompt.strip():
            print("\033[93mWARNING: AIMLAPIFluxGenerator - Prompt is empty.\033[0m")
            return (torch.zeros(1, 3, 64, 64, dtype=torch.float32),)

        api_url = "https://api.aimlapi.com/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {effective_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model_id,
            "prompt": prompt,
            "num_images": 1,
            "output_format": output_format,
            "aspect_ratio": aspect_ratio,
            "safety_tolerance": safety_tolerance,
            "enable_safety_checker": True if enable_safety_checker == "enable" else False,
            "raw": False,
            "seed": seed
        }

        # --- Optional img2img input ---
        if input_image is not None:
            try:
                print("AIMLAPIFluxGenerator: Processing input image...")
                img_np = input_image[0].permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                buffer = io.BytesIO()
                pil_img.save(buffer, format="JPEG")
                b64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
                payload["init_image"] = b64_image
                print("AIMLAPIFluxGenerator: Input image attached to payload.")
            except Exception as e:
                print(f"\033[91mERROR: Failed to process input image: {e}\033[0m")

        print(f"AIMLAPIFluxGenerator: Sending request to {model_id} (Seed: {seed})...")
        print(f"AIMLAPIFluxGenerator: Prompt: \"{prompt[:80]}{'...' if len(prompt) > 80 else ''}\"")

        image_tensor = torch.zeros(1, 3, 64, 64, dtype=torch.float32)
        response = None
        start_time = time.time()

        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=300)
            response.raise_for_status()
            response_data = response.json()
            end_time = time.time()
            print(f"AIMLAPIFluxGenerator: Response received in {end_time - start_time:.2f}s")

            image_b64 = None
            data_source = None
            if isinstance(response_data.get('data'), list) and response_data['data']:
                data_source = response_data['data'][0]
            elif isinstance(response_data.get('images'), list) and response_data['images']:
                data_source = response_data['images'][0]

            if data_source:
                image_b64 = data_source.get('b64_json') or data_source.get('image_data') or data_source.get('image')

            if not image_b64 or not isinstance(image_b64, str):
                print("\033[91mERROR: Could not find valid Base64 image data in API response.\033[0m")
                print(json.dumps(response_data, indent=2))
                return (image_tensor,)

            print("AIMLAPIFluxGenerator: Decoding and converting image...")
            image_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor_hwc = torch.from_numpy(img_np)
            img_tensor_chw = img_tensor_hwc.permute(2, 0, 1)
            image_tensor = img_tensor_chw.unsqueeze(0)

            print(f"AIMLAPIFluxGenerator: Image tensor shape: {image_tensor.shape}")

        except requests.exceptions.HTTPError as e:
            print(f"\033[91mERROR: HTTP Error {e.response.status_code}\033[0m")
            try:
                print("AIMLAPI Error:", json.dumps(e.response.json(), indent=2))
            except Exception:
                print("Raw error response:", e.response.text[:500])
        except requests.exceptions.Timeout:
            print("\033[91mERROR: API request timed out.\033[0m")
        except requests.exceptions.RequestException as e:
            print(f"\033[91mERROR: Network error: {e}\033[0m")
        except Exception as e:
            print(f"\033[91mERROR: Unexpected error: {e}\033[0m")

        return (image_tensor,)

# Register the node with ComfyUI
NODE_CLASS_MAPPINGS = {
    "AIMLAPIFluxGenerator": AIMLAPIFluxGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIMLAPIFluxGenerator": "Image Generator (Flux via AIMLAPI)"
}