import torch
import numpy as np
import fal_client
from PIL import Image
import comfy.utils
import io
import base64
import requests

t2i_model_ids = [
    "fal-ai/hidream-i1-full",
    "fal-ai/hidream-i1-dev",
    "fal-ai/hidream-i1-fast",
    "fal-ai/flux/dev",
]


class FalStreamingTextToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "multiline": False,
                        "default": "",
                        "tooltip": "Your fal.ai API key",
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "a cat holding a skateboard which has 'fal' and 'comfydeploy' written on it in red spray paint",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
                "model_id": (t2i_model_ids,),
                "steps": ("INT", {"default": 50, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "stream_generation"
    CATEGORY = "fal.ai"

    def __init__(self):
        self.stream = None
        self.current_step = 0
        self.total_steps = 0
        self.final_image = None

    def stream_generation(self, api_key, prompt, seed, model_id, steps):
        self.total_steps = steps
        try:
            fal_client.initialize(api_key)
            print("Starting streaming task")

            self.stream = fal_client.stream(
                model_id,
                arguments={
                    "prompt": prompt,
                    "num_inference_steps": steps,
                    "seed": seed,
                },
            )

            self.current_step = 0
            for event in self.stream:
                self.current_step += 1

                if "images" in event:
                    image_data = event["images"][0]
                    url = image_data["url"]

                    if isinstance(url, str) and "data:image" in url:
                        base64_data = url.split(",")[1]
                        image_bytes = base64.b64decode(base64_data)
                        image = Image.open(io.BytesIO(image_bytes))
                    else:
                        response = requests.get(url)
                        image = Image.open(io.BytesIO(response.content))

                    image_np = np.array(image)
                    image_tensor = torch.from_numpy(image_np).float() / 255.0
                    image_tensor = image_tensor.unsqueeze(0)
                    self.final_image = image_tensor

                    pbar = comfy.utils.ProgressBar(self.total_steps)
                    pbar.update_absolute(
                        self.current_step, self.total_steps, ("JPEG", image, None)
                    )

                if self.current_step >= steps:
                    break

            return (self.final_image,)

        except Exception as e:
            print(f"Error in fal.ai streaming: {e}")
            return torch.zeros((1, 512, 512, 3), dtype=torch.float32)


NODE_CLASS_MAPPINGS = {
    "FalStreamingTextToImage": FalStreamingTextToImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FalStreamingTextToImage": "Fal Streaming Text-to-Image"
}