class PromptDeconcatenator:
    """
    A simple ComfyUI node to split a single input text into positive and negative prompts using a delimiter.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt_text": ("STRING", {"multiline": True}),
                "delimiter": ("STRING", {"default": "!!!"}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "split_prompt"
    CATEGORY = "Prompt/Utility"

    def split_prompt(self, prompt_text, delimiter):
        if delimiter and delimiter in prompt_text:
            parts = prompt_text.split(delimiter, 1)
            return parts[0].strip(), parts[1].strip()
        return prompt_text.strip(), ""

NODE_CLASS_MAPPINGS = {
    "PromptDeconcatenator": PromptDeconcatenator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptDeconcatenator": "Prompt Deconcatenator",
}