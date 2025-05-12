import requests
import json
import base64
import os
import io # Needed for handling bytes in memory
import re
import time
import numpy as np
import torch
from PIL import Image # Need Pillow for image manipulation

# Helper function to get API key (prioritize input, then environment variable)
def get_api_key(api_key_input):
    if api_key_input and api_key_input.strip() and api_key_input.strip().lower() not in ["your_aimlapi_key", ""]:
         return api_key_input.strip()
    env_key = os.environ.get('AIMLAPI_KEY')
    if env_key:
        # Optional: print message if using environment variable
        # print("AIMLAPIFluxGenerator: Using AIMLAPI_KEY from environment variable.")
        return env_key
    return None

# --- The ComfyUI Node Class ---

class AIMLAPIFluxGenerator:
    """
    A ComfyUI node to generate images using the AIMLAPI Flux Pro model.
    Takes prompt and parameters, calls the API, and returns the image tensor.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # Valid aspect ratios supported by many models
        # (Add/remove based on Flux specific capabilities if known, these are common ones)
        aspect_ratios = ["16:9", "1:1", "9:16", "4:3", "3:4", "21:9", "9:21"]
        output_formats = ["jpeg", "png"] # As per original script
        safety_tolerances = ["1", "2", "3", "4", "5", "6"] # As per previous info
        boolean_options = ["enable", "disable"]

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Epic cinematic shot of a dragon flying over a futuristic city, dramatic lighting, hyperrealistic, 8k"}),
                "api_key": ("STRING", {"multiline": False, "default": "YOUR_AIMLAPI_KEY"}),
                "aspect_ratio": (aspect_ratios, {"default": "16:9"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), # Allow user input seed
            },
            "optional": {
                 "model_id": ("STRING", {"multiline": False, "default": "flux-pro/v1.1-ultra"}),
                 "output_format": (output_formats, {"default": "jpeg"}),
                 "safety_tolerance": (safety_tolerances, {"default": "1"}),
                 "enable_safety_checker": (boolean_options, {"default": "enable"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "API/AIMLAPI"

    def generate_image(self, prompt, api_key, aspect_ratio, seed, model_id="flux-pro/v1.1-ultra", output_format="jpeg", safety_tolerance="1", enable_safety_checker="enable"):
        """
        The main execution method called by ComfyUI.
        """
        # --- Get API Key ---
        effective_api_key = get_api_key(api_key)
        if not effective_api_key:
             print("\033[91mERROR: AIMLAPIFluxGenerator - AIMLAPI Key not found. Please provide it in the node input or set the AIMLAPI_KEY environment variable.\033[0m")
             # Return a dummy tensor on critical failure to prevent crash
             return (torch.zeros(1, 64, 64, 3, dtype=torch.float32),)

        if not prompt or not prompt.strip():
             print("\033[93mWARNING: AIMLAPIFluxGenerator - Prompt is empty.\033[0m")
             return (torch.zeros(1, 64, 64, 3, dtype=torch.float32),)


        # --- Prepare API Call ---
        api_url = "https://api.aimlapi.com/v1/images/generations"
        headers = {
            "Authorization": f"Bearer {effective_api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model_id,
            "prompt": prompt,
            "num_images": 1, # ComfyUI usually handles batches externally if needed
            "output_format": output_format,
            "aspect_ratio": aspect_ratio,
            "safety_tolerance": safety_tolerance,
            "enable_safety_checker": True if enable_safety_checker == "enable" else False,
            "raw": False # Default from script example
        }
        # Add seed (0 is a valid seed for many APIs)
        payload["seed"] = seed

        print(f"AIMLAPIFluxGenerator: Sending request to {model_id} (Seed: {seed})...")
        # Limit printing potentially long prompt in logs
        print(f"AIMLAPIFluxGenerator: Prompt (start): \"{prompt[:80]}{'...' if len(prompt)>80 else ''}\"")
        start_time = time.time()
        image_tensor = torch.zeros(1, 64, 64, 3, dtype=torch.float32) # Default error tensor
        response = None # Define outside try

        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=300) # 5 min timeout
            response.raise_for_status()

            response_data = response.json()
            end_time = time.time()
            print(f"AIMLAPIFluxGenerator: Response received (took {end_time - start_time:.2f} seconds)")

            # --- Extract Base64 Image Data ---
            image_b64 = None
            data_source = None
            if isinstance(response_data.get('data'), list) and len(response_data['data']) > 0:
                data_source = response_data['data'][0]
            elif isinstance(response_data.get('images'), list) and len(response_data['images']) > 0:
                 data_source = response_data['images'][0]

            if data_source:
                image_b64 = data_source.get('b64_json') or data_source.get('image_data') or data_source.get('image')

            if not image_b64 or not isinstance(image_b64, str):
                 print("\033[91mERROR: AIMLAPIFluxGenerator - Could not find valid Base64 image data in API response.\033[0m")
                 print("--- Response Data ---")
                 try:
                     print(json.dumps(response_data, indent=2))
                 except Exception as print_err:
                      print(f"(Could not print full response: {print_err})")
                      if response: print(f"Raw Response Text: {response.text[:500]}...")
                 print("--- End Response ---")
                 return (image_tensor,) # Return error tensor

            # --- Decode Base64 and Convert to Tensor ---
            print("AIMLAPIFluxGenerator: Decoding Base64 image...")
            try:
                 image_bytes = base64.b64decode(image_b64)
            except (base64.binascii.Error, ValueError) as decode_err:
                 print(f"\033[91mERROR: AIMLAPIFluxGenerator - Failed to decode Base64 string: {decode_err}\033[0m")
                 return (image_tensor,) # Return error tensor


            print("AIMLAPIFluxGenerator: Converting image data to tensor...")
            # Load image from bytes using Pillow
            try:
                img = Image.open(io.BytesIO(image_bytes))
                img_rgb = img.convert("RGB") # Ensure RGB format

                # Convert to NumPy array and normalize to [0, 1]
                img_np = np.array(img_rgb).astype(np.float32) / 255.0

                # Convert to PyTorch tensor -> shape (H, W, C)
                img_tensor_hwc = torch.from_numpy(img_np)

                # Add batch dimension -> shape (1, H, W, C) - ComfyUI standard
                image_tensor = img_tensor_hwc.unsqueeze(0)
                print(f"AIMLAPIFluxGenerator: Image tensor created successfully (shape: {image_tensor.shape})")

            except Exception as img_err:
                print(f"\033[91mERROR: AIMLAPIFluxGenerator - Failed to process image bytes: {img_err}\033[0m")
                return (image_tensor,) # Return error tensor

        # --- Error Handling for API Call ---
        except requests.exceptions.HTTPError as e:
            print(f"\033[91mERROR: AIMLAPIFluxGenerator - HTTP Error {e.response.status_code} calling AIMLAPI.\033[0m")
            try:
                error_details = e.response.json()
                print("AIMLAPI Error Response:", json.dumps(error_details, indent=2))
            except json.JSONDecodeError:
                print("AIMLAPI Error Response (non-JSON):", e.response.text[:500] + "...")
            except Exception as print_err:
                 print(f"(Could not display full error response: {print_err})")
                 if response: print(f"Raw Response Text: {response.text[:500]}...")

        except requests.exceptions.Timeout:
            print(f"\033[91mERROR: AIMLAPIFluxGenerator - API request timed out.\033[0m")
        except requests.exceptions.RequestException as e:
            print(f"\033[91mERROR: AIMLAPIFluxGenerator - Network or request error: {e}\033[0m")
        except Exception as e: # Catch any other unexpected errors
            print(f"\033[91mERROR: AIMLAPIFluxGenerator - An unexpected node error occurred: {e}\033[0m")
            import traceback
            traceback.print_exc()


        # Return the resulting tensor (or the error tensor if something failed)
        return (image_tensor,)


# --- ComfyUI Registration ---

NODE_CLASS_MAPPINGS = {
    "AIMLAPIFluxGenerator": AIMLAPIFluxGenerator # Class name should be globally unique
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIMLAPIFluxGenerator": "Image Generator (Flux via AIMLAPI)" # Friendly name for the UI
}