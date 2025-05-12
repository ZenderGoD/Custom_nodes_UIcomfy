# Required standard libraries
import time
import io
import base64
import os
from PIL import Image
import torch
import numpy as np
from fal import Client

print("Loading FAL_AURA.py...")  # Debug print

# Assumes these utility functions are available (e.g., in ../utils.py)
try:
    from ..utils import pil2tensor, tensor2pil
except ImportError:
    print("Warning: ComfyUI utils not found in parent directory. Node may not function correctly.")
    def pil2tensor(images):
         print("Dummy pil2tensor called")
         return images
    def tensor2pil(tensors):
         print("Dummy tensor2pil called")
         return tensors

class FalAuraSR:
    """
    A ComfyUI node for upscaling images using Fal's Aura-SR API.
    Uses the queue-based API workflow (submit, poll status, get result).
    """
    
    def __init__(self):
        print("Initializing FalAuraSR node...")  # Debug print
        self.client = None

    @classmethod
    def INPUT_TYPES(s):
        print("Getting input types for FalAuraSR...")  # Debug print
        return {
            "required": {
                "image": ("IMAGE",),
                "fal_api_key": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": os.getenv("FAL_KEY", ""),
                        "tooltip": "Your FAL API Key (can be set via FAL_KEY env var). KEEP SECRET!"
                    },
                ),
            },
            "optional": {
                "upscaling_factor": (
                    "INT",
                    {
                        "default": 4,
                        "min": 4,
                        "max": 4,
                        "step": 1,
                        "display": "number",
                        "tooltip": "Upscaling factor. Only 4x is currently supported by the API."
                    },
                ),
                "overlapping_tiles": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Whether to use overlapping tiles for upscaling. Setting this to true helps remove seams but increases inference time."
                    },
                ),
                "checkpoint": (
                    ["v1", "v2"],
                    {
                        "default": "v2",
                        "tooltip": "Checkpoint model to use ('v1' or 'v2')."
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upscaled_image",)
    FUNCTION = "execute"
    CATEGORY = "image/upscale"
    OUTPUT_NODE = False

    def execute(self, image, fal_api_key, upscaling_factor=4, overlapping_tiles=True, checkpoint="v2"):
        print("Executing FalAuraSR node...")  # Debug print
        # --- Input Validation ---
        if not fal_api_key:
            raise ValueError("FAL API Key is required.")

        # Initialize the fal.ai client
        if not self.client:
            self.client = Client(key=fal_api_key)

        # --- Data Preparation ---
        # Convert tensor to PIL Image
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # If batch dimension exists
                image = image[0]  # Take first image from batch
            if image.dim() == 3 and image.shape[0] == 3:  # If CHW format
                image = image.permute(1, 2, 0)  # Convert to HWC
            
            # Convert to numpy and then to PIL
            image_np = (image.cpu().numpy() * 255).astype('uint8')
            image_pil = Image.fromarray(image_np).convert('RGB')
        else:
            raise ValueError(f"Expected torch.Tensor input, got {type(image)}")

        # Convert PIL Image to Base64 Data URI
        buffer = io.BytesIO()
        image_pil.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        data_uri = f"data:image/png;base64,{encoded_image}"

        try:
            # Use the fal.ai client to submit the request
            result = self.client.submit(
                "fal-ai/aura-sr",
                {
                    "input": {
                        "image_url": data_uri,
                        "upscaling_factor": upscaling_factor,
                        "overlapping_tiles": overlapping_tiles,
                        "checkpoint": checkpoint,
                    }
                }
            )

            # Wait for the result
            print("Waiting for FAL Aura-SR processing...")
            while True:
                status = result.status()
                if status.status == "COMPLETED":
                    break
                elif status.status == "IN_PROGRESS":
                    # Print any logs
                    for log in status.logs:
                        print(f"FAL Log: {log.message}")
                    time.sleep(1)
                elif status.status in ["FAILED", "CANCELLED"]:
                    raise RuntimeError(f"FAL Aura-SR job failed or was cancelled. Status: {status.status}. Error: {status.error}")
                else:
                    raise RuntimeError(f"Unexpected status: {status.status}")

            # Get the result
            result_data = result.get()
            image_url = result_data["image"]["url"]

            # Download the upscaled image
            print("Downloading upscaled image...")
            response = self.client.http.get(image_url)
            upscaled_image_pil = Image.open(io.BytesIO(response.content)).convert("RGB")
            print("Upscaled image downloaded successfully.")

            # Convert back to tensor format
            upscaled_image_np = np.array(upscaled_image_pil)
            upscaled_image_tensor = torch.from_numpy(upscaled_image_np).float() / 255.0
            upscaled_image_tensor = upscaled_image_tensor.permute(2, 0, 1).unsqueeze(0)

            return (upscaled_image_tensor,)

        except Exception as e:
            raise RuntimeError(f"FAL Aura-SR processing failed: {str(e)}")

    @classmethod
    def IS_CHANGED(s, image, fal_api_key, upscaling_factor, overlapping_tiles, checkpoint):
        """
        Force the node to re-execute if any of the inputs change.
        """
        return ""

print("Registering FalAuraSR node...")  # Debug print

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "FalAuraSR": FalAuraSR
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FalAuraSR": "FAL Aura-SR Upscaler"
}

print("FAL_AURA.py loaded successfully!")  # Debug print