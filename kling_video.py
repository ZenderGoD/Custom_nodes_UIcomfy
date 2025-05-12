import requests
import time
import os
import tempfile
from PIL import Image
import numpy as np
import io

class KlingAIMLAPIVideoGenerator:
    """
    A ComfyUI node to generate a video from an image tensor using AIMLAPI Kling model.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Mona Lisa puts on glasses with her hands."}),
                "image": ("IMAGE",),  # Now accepts ComfyUI image input
                "duration": ("STRING", {"default": "5"}),
                "api_key": ("STRING", {"default": os.environ.get("AIMLAPI_KEY", "YOUR_AIMLAPI_KEY")}),
            }
        }

    RETURN_TYPES = ("STRING",)  # returns video URL
    RETURN_NAMES = ("video_url",)
    FUNCTION = "generate_video"
    CATEGORY = "API/AIMLAPI"

    def tensor_to_pil(self, image_tensor):
        image_array = (image_tensor[0].numpy() * 255).astype(np.uint8)
        return Image.fromarray(image_array)

    def generate_video(self, prompt, image, duration, api_key):
        base_url = "https://api-staging.aimlapi.com/v2"
        headers = {
            "Authorization": f"Bearer {api_key}",
        }

        # Convert tensor to image and save to temp buffer
        pil_image = self.tensor_to_pil(image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Upload image with form data
        files = {
            "image_file": ("input.jpg", buffer, "image/jpeg"),
        }
        data = {
            "model": "klingai/v2-master-image-to-video",
            "prompt": prompt,
            "duration": duration,
        }

        print("KlingAIMLAPIVideoGenerator: Submitting generation request...")
        submit_response = requests.post(
            f"{base_url}/generate/video/kling/generation",
            data=data,
            files=files,
            headers=headers
        )

        if submit_response.status_code >= 400:
            print("=== AIMLAPI Video Submit Error ===")
            print(f"Status Code: {submit_response.status_code}")
            print("Response Text:", submit_response.text)
            print("Request Headers:", submit_response.request.headers)
            print("Request Body:", submit_response.request.body)
            print("==================================")
            return (f"[Error: {submit_response.text}]",)

        gen_id = submit_response.json().get("id")
        print(f"Generation ID: {gen_id}")

        if not gen_id:
            return ("[Failed to retrieve generation ID]",)

        # Polling
        timeout = 600
        poll_interval = 10
        start_time = time.time()

        print("Polling for video generation...")
        while time.time() - start_time < timeout:
            poll_response = requests.get(
                f"{base_url}/generate/video/kling/generation",
                headers={"Authorization": f"Bearer {api_key}"},
                params={"generation_id": gen_id},
            )
            result = poll_response.json()
            status = result.get("status", "unknown")
            print(f"Status: {status}")

            if status == "completed":
                video_info = result.get("video", {})
                video_url = video_info.get("url", "[No video URL returned]")
                print("Video ready:", video_url)
                return (video_url,)
            elif status in {"failed", "canceled"}:
                return (f"[Generation {status}]",)

            time.sleep(poll_interval)

        return ("[Generation timeout]",)


# ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "KlingAIMLAPIVideoGenerator": KlingAIMLAPIVideoGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KlingAIMLAPIVideoGenerator": "Video Generator (Kling via AIMLAPI)",
}