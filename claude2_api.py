import base64
import io
import os
from PIL import Image
import torch
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

class ClaudeImagePromptFormatter:
    """
    ComfyUI node to format image(s) + prompt into a descriptive Claude-generated positive/negative prompt string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True}),
                "delimiter": ("STRING", {"default": "###"}),
                "api_key": ("STRING", {"default": os.environ.get("ANTHROPIC_API_KEY", "")}),
            },
            "optional": {
                "image_2": ("IMAGE",),
                "job_descriptor": ("STRING", {"multiline": True, "default": ""})
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "format_prompt"
    CATEGORY = "Prompt/Formatter"

    def tensor_to_base64_image(self, image_tensor):
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.ndim == 4:
                image_tensor = image_tensor.squeeze(0)
            if image_tensor.ndim == 3 and image_tensor.shape[0] in [1, 3]:
                image_np = image_tensor.cpu().numpy()
                image_np = (image_np * 255).clip(0, 255).astype("uint8")
                image_np = image_np.transpose(1, 2, 0)
            elif image_tensor.ndim == 3:
                image_np = image_tensor.cpu().numpy()
                image_np = image_np.squeeze()
                image_np = (image_np * 255).clip(0, 255).astype("uint8")
            else:
                raise ValueError("Unsupported tensor shape")

            image_pil = Image.fromarray(image_np)
            buffered = io.BytesIO()
            image_pil.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return img_str, "image/png"
        else:
            raise TypeError("Expected image tensor as torch.Tensor")

    def format_prompt(self, image_1, prompt, delimiter, api_key, image_2=None, job_descriptor=""):
        image1_data, image1_media_type = self.tensor_to_base64_image(image_1)
        content = [
            {"type": "text", "text": "Image 1:"},
            {"type": "image", "source": {"type": "base64", "media_type": image1_media_type, "data": image1_data}},
        ]

        if image_2 is not None:
            image2_data, image2_media_type = self.tensor_to_base64_image(image_2)
            content += [
                {"type": "text", "text": "Image 2:"},
                {"type": "image", "source": {"type": "base64", "media_type": image2_media_type, "data": image2_data}},
            ]

        final_prompt = "".join([
            job_descriptor + "\n" if job_descriptor else "",
            prompt,
        ])

        content.append({"type": "text", "text": final_prompt})

        client = Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-3-7-sonnet-20250219",
            max_tokens=1024,
            messages=[{"role": "user", "content": content}]
        )

        response_text = message.content[0].text.strip()

        if delimiter and delimiter in response_text:
            parts = response_text.split(delimiter, 1)
            return parts[0].strip(), parts[1].strip()
        else:
            return response_text, ""

NODE_CLASS_MAPPINGS = {
    "ClaudeImagePromptFormatter": ClaudeImagePromptFormatter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClaudeImagePromptFormatter": "Claude Image Prompt Formatter",
}
