import torch
import io
import json
import requests
from PIL import Image
from typing import Any

# Define dummy IO types for ComfyUI compatibility
class IO:
    STRING = "STRING"
    INT = "INT"
    IMAGE = "IMAGE"
    MASK = "MASK"
    COMBO = "COMBO"

class SelfOpenAIImage:
    """
    Generates or edits images using OpenAI GPT-Image-1 API.
    """

    def __init__(self):
        self.api_url = "https://api.openai.com/v1/images"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (IO.STRING, {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Prompt for image generation"
                }),
                "api_key": (IO.STRING, {
                    "default": "",
                    "tooltip": "OpenAI API Key"
                }),
            },
            "optional": {
                "seed": (IO.INT, {"default": 0, "tooltip": "Seed (not used yet)"}),
                "quality": (IO.COMBO, {
                    "options": ["low", "medium", "high"],
                    "default": "low",
                    "tooltip": "Image quality"
                }),
                "background": (IO.COMBO, {
                    "options": ["opaque", "transparent"],
                    "default": "opaque",
                    "tooltip": "Background type"
                }),
                "size": (IO.COMBO, {
                    "options": ["auto", "1024x1024", "1024x1536", "1536x1024"],
                    "default": "1024x1024",
                    "tooltip": "Output image size"
                }),
                "n": (IO.INT, {"default": 1, "min": 1, "max": 4, "tooltip": "Number of images"}),
                "image": (IO.IMAGE, {"default": None, "tooltip": "Optional image input"}),
                "mask": (IO.MASK, {"default": None, "tooltip": "Optional inpainting mask"}),
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    FUNCTION = "api_call"
    CATEGORY = "api/image"
    DESCRIPTION = "OpenAI GPT-Image-1 generator"

    def tensor_to_image_bytes(self, tensor):
        tensor = tensor.squeeze()
        img = Image.fromarray((tensor.numpy() * 255).astype("uint8"))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        return img_byte_arr

    def downscale_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 4:
            tensor = tensor[0]
        tensor = tensor.permute(1, 2, 0).cpu().clamp(0, 1)
        return tensor

    def api_call(
        self,
        prompt,
        api_key,
        seed=0,
        quality="low",
        background="opaque",
        size="1024x1024",
        n=1,
        image=None,
        mask=None,
    ):
        if not api_key:
            raise Exception("API key is required")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        is_edit = image is not None
        files = {}
        data = {
            "model": "gpt-image-1",
            "prompt": prompt,
            "n": n,
            "quality": quality,
            "background": background,
            "size": size,
            "seed": seed,
        }

        # Build full endpoint
        endpoint = f"{self.api_url.rstrip('/')}/{'edits' if is_edit else 'generations'}"

        if is_edit:
            img_bytes = self.tensor_to_image_bytes(self.downscale_tensor(image))
            img_bytes.name = "image.png"
            files["image"] = img_bytes

            if mask is not None:
                mask_tensor = mask.squeeze().cpu()
                rgba = torch.zeros(mask_tensor.shape[0], mask_tensor.shape[1], 4)
                rgba[:, :, 3] = 1 - mask_tensor
                mask_img = Image.fromarray((rgba.numpy() * 255).astype("uint8"))
                mask_io = io.BytesIO()
                mask_img.save(mask_io, format="PNG")
                mask_io.seek(0)
                mask_io.name = "mask.png"
                files["mask"] = mask_io

            response = requests.post(endpoint, headers=headers, data=data, files=files)
        else:
            response = requests.post(endpoint, headers=headers, json=data)

        if not response.ok:
            raise Exception(f"API Error: {response.status_code} - {response.text}")

        json_data = response.json()
        image_urls = json_data.get("data", [])
        if not image_urls:
            raise Exception("No images returned from API")

        result_images = []
        for img_data in image_urls:
            if isinstance(img_data, dict) and "url" in img_data:
                image_response = requests.get(img_data["url"])
                img = Image.open(io.BytesIO(image_response.content)).convert("RGB")
                img_tensor = torch.tensor(np.array(img)).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
                result_images.append(img_tensor)
            else:
                raise Exception("Unexpected image format in API response")

        return (torch.cat(result_images, dim=0),)

# Node mapping
NODE_CLASS_MAPPINGS = {
    "SelfOpenAIImage": SelfOpenAIImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelfOpenAIImage": "OpenAI GPT-Image-1 (Custom)",
}