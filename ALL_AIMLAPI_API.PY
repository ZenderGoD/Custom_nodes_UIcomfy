import requests
import time

class AIMLAPIUniversalNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([
                    "flux-pro/v1.1-ultra",
                    "claude-3-5-sonnet-latest",
                    "klingai/v2-master-image-to-video"
                ],),
                "api_key": ("STRING", {"multiline": False, "default": "74a04a818c4543e48a0fcc01afb33c9c"}),
                "input_prompt": ("STRING", {"multiline": True, "default": "Describe your scene here."}),
                "image_url": ("STRING", {"multiline": False, "default": ""}),
                "duration": ("STRING", {"multiline": False, "default": "5"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "run"
    CATEGORY = "AIMLAPI"

    def run(self, model, api_key, input_prompt, image_url, duration):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        if model == "flux-pro/v1.1-ultra":
            response = requests.post(
                "https://api.aimlapi.com/v1/images/generations",
                headers=headers,
                json={"prompt": input_prompt, "model": model}
            )
            response.raise_for_status()
            url = response.json()["images"][0]["url"]
            return (url,)

        elif model == "claude-3-5-sonnet-latest":
            response = requests.post(
                "https://api.aimlapi.com/v1/chat/completions",
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": input_prompt}]
                }
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            return (content,)

        elif model == "klingai/v2-master-image-to-video":
            base_url = "https://api-staging.aimlapi.com/v2"
            response = requests.post(
                f"{base_url}/generate/video/kling/generation",
                headers=headers,
                json={
                    "model": model,
                    "prompt": input_prompt,
                    "image_url": image_url,
                    "duration": duration
                }
            )
            response.raise_for_status()
            gen_id = response.json().get("id")

            if not gen_id:
                return ("Failed to get generation ID",)

            timeout = 300
            start_time = time.time()

            while time.time() - start_time < timeout:
                status_response = requests.get(
                    f"{base_url}/generate/video/kling/generation",
                    headers=headers,
                    params={"generation_id": gen_id}
                )
                status_data = status_response.json()
                status = status_data.get("status")

                if status in ["waiting", "active", "queued", "generating"]:
                    time.sleep(10)
                elif status == "completed":
                    return (status_data["video"]["url"],)
                else:
                    return (f"Failed: {status}",)

            return ("Timeout waiting for video",)

        return ("Unsupported model",)


NODE_CLASS_MAPPINGS = {
    "AIMLAPIUniversalNode": AIMLAPIUniversalNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AIMLAPIUniversalNode": "AIMLAPI Universal Node"
}
