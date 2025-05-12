class VideoLinkDisplayNode:
    """
    Displays a clickable video link in the ComfyUI UI output panel.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_url": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_link",)
    OUTPUT_NODE = True
    CATEGORY = "Display"

    FUNCTION = "display_link"

    def display_link(self, video_url):
        html_link = f'<a href="{video_url}" target="_blank" rel="noopener noreferrer">▶️ Click to view/download video</a>'
        return (html_link,)


NODE_CLASS_MAPPINGS = {
    "KlingAIMLAPIVideoGenerator": KlingAIMLAPIVideoGenerator,
    "VideoLinkDisplayNode": VideoLinkDisplayNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KlingAIMLAPIVideoGenerator": "Video Generator (Kling via AIMLAPI)",
    "VideoLinkDisplayNode": "Show Video Link",
}