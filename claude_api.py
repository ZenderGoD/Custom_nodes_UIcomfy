import os
import base64
import mimetypes
from anthropic import Anthropic
from anthropic.types import Message

class ClaudePromptFromImages:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "user_prompt": ("STRING", {"multiline": True, "default": "Describe the images in detail as a prompt for image generation. Include both positive and negative aspects. Use '###' to separate them."}),
                "delimiter": ("STRING", {"default": "###"}),
                "api_key": ("STRING", {"default": os.getenv("ANTHROPIC_API_KEY", "sk-ant-...")}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "generate_prompt"
    CATEGORY = "API/Claude"

    def image_to_base64(self, image):
        # Assumes 'image' is a ComfyUI tensor of shape [C, H, W]
        import io
        from PIL import Image
        import numpy as np

        # Convert tensor to image
        np_img = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
        if np_img.shape[0] == 1:
            np_img = np_img.squeeze(0)
        elif np_img.shape[0] == 3:
            np_img = np.transpose(np_img, (1, 2, 0))

        image_pil = Image.fromarray(np_img)
        buf = io.BytesIO()
        image_pil.save(buf, format="PNG")
        image_bytes = buf.getvalue()
        base64_str = base64.b64encode(image_bytes).decode("utf-8")
        return base64_str, "image/png"

    def generate_prompt(self, image_1, image_2, user_prompt, delimiter, api_key):
        base64_1, mime1 = self.image_to_base64(image_1)
        base64_2, mime2 = self.image_to_base64(image_2)

        client = Anthropic(api_key=api_key)

        try:
            msg: Message = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Image 1:"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime1,
                                    "data": base64_1,
                                },
                            },
                            {"type": "text", "text": "Image 2:"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime2,
                                    "data": base64_2,
                                },
                            },
                            {"type": "text", "text": user_prompt}
                        ]
                    }
                ]
            )

            text = msg.content[0].text.strip()
            if delimiter in text:
                positive, negative = text.split(delimiter, 1)
                return positive.strip(), negative.strip()
            else:
                return text.strip(), "[Delimiter not found – provide a clearer instruction or delimiter.]"

        except Exception as e:
            return "[Error communicating with Claude: {}]".format(str(e)), ""

# ComfyUI registration
NODE_CLASS_MAPPINGS = {
    "ClaudePromptFromImages": ClaudePromptFromImages
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClaudePromptFromImages": "Prompt Generator (Claude + Images)"
}