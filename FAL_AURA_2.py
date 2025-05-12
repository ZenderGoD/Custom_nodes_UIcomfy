import torch
import numpy as np
import requests
import io
import PIL.Image
import os
import asyncio

# Make sure to install the necessary libraries:
# pip install fal_client requests Pillow numpy torch

# The fal client is designed with async in mind, so we'll need to use async/await.
# ComfyUI nodes are typically sync, but the backend can handle async node functions.
# We import the async version of the client if available, otherwise fall back.
try:
    # Use the async client if available (Fal 0.4.0+)
    from fal import Doinfo, models, apps, subscribe, queue, storage
    fal_client = Doinfo(key=os.environ.get("FAL_KEY")) # Initialize without key, will check config/env later
    # Manually assign the async functions we need from the initialized client instance
    fal_subscribe = fal_client.subscribe
    fal_queue = fal_client.queue
    fal_storage = fal_client.storage
    fal_config = fal_client.config
    print("Fal.ai Upscaler: Using fal_client instance methods.")
except ImportError:
    # Fallback for older versions of the client or if Doinfo is not found
    # This might use global state or require manual config.
    # This path is less preferred but included for compatibility if needed.
    # Note: The older client structure is less clear about async handling per function.
    # If you encounter issues with older versions, ensure fal_client is properly configured for async.
    try:
        from fal import subscribe, queue, storage, config
        fal_subscribe = subscribe
        fal_queue = queue
        fal_storage = storage
        fal_config = config
        print("Fal.ai Upscaler: Using direct fal module functions (potentially older client).")
    except ImportError as e:
         print(f"Fal.ai Upscaler: Error importing fal client. Please install it: pip install fal_client")
         raise e # Re-raise to fail early if client isn't installed


# Helper function to convert ComfyUI tensor to PIL Image
def tensor_to_pil(image_tensor):
    # image_tensor is [B, H, W, C], float [0, 1]
    batch_size, height, width, channels = image_tensor.shape
    if batch_size > 1:
        print("Warning: Fal.ai Upscaler node processing only the first image in the batch.")
    img_np = image_tensor[0].cpu().numpy() # Take first image, convert to numpy
    img_np = np.clip(img_np * 255., 0., 255.).astype(np.uint8) # Scale and convert to uint8

    if channels == 4:
        # PIL needs RGBA
        return PIL.Image.fromarray(img_np, 'RGBA')
    elif channels == 3:
         return PIL.Image.fromarray(img_np, 'RGB')
    else:
         # Try to convert to RGB
         print(f"Warning: Unsupported image channels ({channels}), attempting conversion to RGB.")
         return PIL.Image.fromarray(img_np.squeeze(), 'RGB') # Squeeze might be needed for grayscale [H, W, 1]

# Helper function to convert PIL Image to ComfyUI tensor
def pil_to_tensor(image_pil):
    # image_pil is PIL Image
    # Ensure image has an alpha channel for ComfyUI compatibility if it's RGB
    if image_pil.mode == 'RGB':
        image_pil = image_pil.convert('RGBA')
    elif image_pil.mode == 'L': # Grayscale
        image_pil = image_pil.convert('RGBA') # Convert to RGBA
    elif image_pil.mode != 'RGBA':
         # Attempt conversion to RGBA for other modes like P
        try:
             image_pil = image_pil.convert('RGBA')
        except Exception as e:
             print(f"Warning: Could not convert PIL image mode {image_pil.mode} to RGBA: {e}")
             # Fallback to RGB or leave as is? Let's just convert to numpy and hope for the best
             pass # Continue with the original mode numpy conversion

    img_np = np.array(image_pil).astype(np.float32) / 255.0 # Convert to numpy and scale [0, 1]

    # img_np is now [H, W, C]
    img_tensor = torch.from_numpy(img_np)[None,] # Add batch dimension [1, H, W, C]
    return img_tensor


class FalaiUpscaler:
    """
    A custom node to perform image upscaling using the Fal.ai 'fal-ai/aura-sr' API.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # Fal.ai API parameters based on provided documentation
        upscaling_factor_enums = ["4"] # Currently only 4x is available
        checkpoint_enums = ["v1", "v2"] # Possible checkpoint models

        return {
            "required": {
                "image": ("IMAGE",),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "Leave empty to use FAL_KEY env var",
                    "display": "text"
                }),
                "upscaling_factor": (upscaling_factor_enums, {"default": "4"}),
                "overlapping_tiles": (["True", "False"], {"default": "False"}),
                "checkpoint": (checkpoint_enums, {"default": "v1"}),
            },
        }

    # We return a single upscaled image tensor
    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("upscaled_image",) # Optional, defaults to RETURN_TYPES names

    FUNCTION = "upscale_image_async" # Define the async function name

    CATEGORY = "Fal.ai" # Category in the ComfyUI menu

    # OUTPUT_NODE = False # Default is False

    async def upscale_image_async(self, image, api_key, upscaling_factor, overlapping_tiles, checkpoint):
        """
        Main function to handle the upscaling process using Fal.ai API.
        Note: This is an async function.
        """
        print(f"Fal.ai Upscaler: Starting upscaling process...")

        # --- 1. Configure Fal.ai Client ---
        # Use input API key if provided, otherwise rely on environment variable
        if api_key:
            print("Fal.ai Upscaler: Using API key from node input.")
            fal_config(credentials=api_key)
        elif os.environ.get("FAL_KEY"):
             print("Fal.ai Upscaler: Using API key from FAL_KEY environment variable.")
             # No need to call fal_config if FAL_KEY is set as it's the default
        else:
            print("Fal.ai Upscaler: ERROR - FAL_KEY environment variable not set and API key not provided in node input.")
            raise ValueError("Fal.ai API Key is required. Set FAL_KEY environment variable or provide it in the node input.")

        # --- 2. Convert and Upload Image ---
        try:
            print("Fal.ai Upscaler: Converting image tensor to PIL and uploading...")
            img_pil = tensor_to_pil(image)

            # Save PIL image to a BytesIO buffer in PNG format
            buffer = io.BytesIO()
            img_pil.save(buffer, format='PNG')
            buffer.seek(0) # Go back to the start of the buffer

            # Upload the image buffer
            # fal_storage.upload expects bytes or a file-like object.
            # Let's try passing the buffer bytes directly with content_type and name.
            # The documentation for js client showed File object, but python client might differ slightly.
            # If getvalue() doesn't work, you might need to pass the buffer object itself
            # or write to a temporary file. Let's try getvalue() first.
            # Let's add a simple filename.
            file_to_upload = buffer.getvalue()
            uploaded_url = await fal_storage.upload(file_to_upload, content_type="image/png", file_name="input.png")

            print(f"Fal.ai Upscaler: Image uploaded successfully. URL: {uploaded_url}")

        except Exception as e:
            print(f"Fal.ai Upscaler: ERROR during image conversion or upload: {e}")
            raise e # Re-raise the exception for ComfyUI to catch

        # --- 3. Call Fal.ai Upscaling API ---
        try:
            print(f"Fal.ai Upscaler: Calling fal-ai/aura-sr model...")

            # Convert string inputs to correct types expected by the API schema
            upscaling_factor_int = int(upscaling_factor) # Schema expects integer
            overlapping_tiles_bool = overlapping_tiles == "True" # Schema expects boolean

            # Prepare input payload
            input_payload = {
                "image_url": uploaded_url,
                "upscaling_factor": upscaling_factor_int,
                "overlapping_tiles": overlapping_tiles_bool,
                "checkpoint": checkpoint,
            }

            # Define the callback for logs (optional, but useful for debugging)
            def log_callback(update):
                if update.status == "IN_PROGRESS":
                     for log in update.logs:
                         print(f"Fal.ai Log: {log.message}")
                elif update.status in ["COMPLETED", "FAILED", "CANCELLED"]:
                     print(f"Fal.ai Status: {update.status}")


            # Use fal.subscribe for a blocking call that waits for the result
            result = await fal_subscribe(
                "fal-ai/aura-sr",
                input=input_payload,
                logs=True,
                onQueueUpdate=log_callback # Pass the log callback
            )

            print(f"Fal.ai Upscaler: API request completed.")
            # print(f"Fal.ai Result Data: {result.data}") # Log full result data for inspection
            print(f"Fal.ai Request ID: {result.requestId}")

            if 'data' not in result or 'image' not in result['data'] or 'url' not in result['data']['image']:
                 print(f"Fal.ai Upscaler: ERROR - Unexpected API result format.")
                 print(f"Full result: {result}")
                 raise ValueError("Fal.ai API returned result in unexpected format.")

            output_image_url = result['data']['image']['url']
            print(f"Fal.ai Upscaler: Output image URL: {output_image_url}")

        except Exception as e:
            print(f"Fal.ai Upscaler: ERROR during Fal.ai API call: {e}")
            # Consider checking specific fal_client exceptions for more detailed error handling
            raise e # Re-raise the exception

        # --- 4. Download and Convert Output Image ---
        try:
            print("Fal.ai Upscaler: Downloading output image...")
            response = requests.get(output_image_url)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # Read the image content into a BytesIO buffer
            output_buffer = io.BytesIO(response.content)
            output_buffer.seek(0)

            # Open the image with PIL
            output_pil = PIL.Image.open(output_buffer)
            print(f"Fal.ai Upscaler: Downloaded image format: {output_pil.format}, mode: {output_pil.mode}")

            # Convert PIL image back to ComfyUI tensor format
            output_tensor = pil_to_tensor(output_pil)
            print("Fal.ai Upscaler: Output image converted to tensor.")

        except requests.exceptions.RequestException as e:
             print(f"Fal.ai Upscaler: ERROR during output image download: {e}")
             raise e # Re-raise request errors
        except Exception as e:
            print(f"Fal.ai Upscaler: ERROR during output image processing: {e}")
            raise e # Re-raise other processing errors


        # --- 5. Return Result ---
        print("Fal.ai Upscaler: Upscaling process finished successfully.")
        return (output_tensor,)

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "FalaiUpscaler": FalaiUpscaler
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "FalaiUpscaler": "Fal.ai Upscaler"
}