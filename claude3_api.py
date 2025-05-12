import base64
import io
import os
from PIL import Image
import numpy as np
from anthropic import Anthropic

class ClaudeImagePromptFormatter:
    """
    A ComfyUI node that takes up to 2 images and an optional job descriptor and returns a formatted prompt using Claude API.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "job_descriptor": ("STRING", {"multiline": True, "default": ""}),
                "api_key": ("STRING", {"default": os.environ.get("CLAUDE_API_KEY", "")} ),
            },
            "optional": {
                "image_2": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("formatted_prompt",)
    FUNCTION = "format_prompt"
    CATEGORY = "Prompt/Claude"

    def _image_to_base64(self, image_tensor):
        image_array = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
        image = Image.fromarray(image_array)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        return img_b64, "image/png"

    def format_prompt(self, image_1, job_descriptor, api_key, image_2=None):
        if not api_key:
            return ("[Missing API Key]",)

        client = Anthropic(api_key=api_key)

        image_1_data, image_1_type = self._image_to_base64(image_1)
        content = [
            {"type": "text", "text": "Image 1:"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_1_type,
                    "data": image_1_data,
                },
            }
        ]

        if image_2 is not None:
            image_2_data, image_2_type = self._image_to_base64(image_2)
            content.extend([
                {"type": "text", "text": "Image 2:"},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_2_type,
                        "data": image_2_data,
                    },
                }
            ])

        # Add optional job descriptor
        instruction_text = job_descriptor.strip() if job_descriptor.strip() else "Describe the images in detailed prompt format for image generation, with positive and negative prompts."
        content.append({"type": "text", "text": instruction_text})

        # Call Claude
        try:
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ]
            )
        except Exception as e:
            return (f"[Claude API error: {str(e)}]",)

        response_text = response.content[0].text.strip()
        return (response_text,)


NODE_CLASS_MAPPINGS = {
    "ClaudeImagePromptFormatter": ClaudeImagePromptFormatter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClaudeImagePromptFormatter": "Claude Image Prompt Formatter",
}
