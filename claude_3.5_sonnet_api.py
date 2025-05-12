import requests
import json
import re
import os # Used for potential environment variable fallback

# Helper function to get API key (prioritize input, then environment variable)
# You might want to customize how the API key is handled for better security
def get_api_key(api_key_input):
    if api_key_input and api_key_input.strip() != "YOUR_AIMLAPI_KEY":
         return api_key_input.strip()
    env_key = os.environ.get('AIMLAPI_KEY')
    if env_key:
        # Optional: print a message if using environment variable
        # print("ClaudePromptEnhancer: Using AIMLAPI_KEY from environment variable.")
        return env_key
    return None

class ClaudePromptEnhancer:
    """
    A ComfyUI node to enhance a simple prompt using Claude 3.5 Sonnet via AIMLAPI.
    Takes a user prompt and returns enhanced positive and negative prompts suitable
    for text-to-image models.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input fields for the node.
        """
        return {
            "required": {
                "user_prompt": ("STRING", {"multiline": True, "default": "A cute cat playing with a ball of yarn"}),
                "api_key": ("STRING", {"multiline": False, "default": "YOUR_AIMLAPI_KEY"}), # Input API Key field
            },
            "optional": {
                "model_id": ("STRING", {"multiline": False, "default": "anthropic/claude-3.5-sonnet"}),
                "max_tokens": ("INT", {"default": 1024, "min": 50, "max": 4096, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    # Output types: Two strings for the prompts
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")

    FUNCTION = "enhance_prompt"

    CATEGORY = "API/Claude" # Or choose/create your preferred category

    def enhance_prompt(self, user_prompt, api_key, model_id="anthropic/claude-3.5-sonnet", max_tokens=1024, temperature=0.7):
        """
        The main execution method called by ComfyUI.
        """
        # --- Get API Key (handles input field vs environment variable) ---
        effective_api_key = get_api_key(api_key)
        if not effective_api_key:
             print("ERROR: ClaudePromptEnhancer - AIMLAPI Key not found. Please provide it in the node input or set the AIMLAPI_KEY environment variable.")
             return ("", "") # Return empty strings on error

        if not user_prompt or not user_prompt.strip():
             print("WARNING: ClaudePromptEnhancer - User prompt is empty.")
             return ("", "")

        # --- Prepare API Call ---
        api_url = "https://api.aimlapi.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {effective_api_key}",
            "Content-Type": "application/json",
        }

        # System prompt to guide Claude
        system_prompt = """You are an expert prompt engineer specializing in creating prompts for advanced text-to-image models. Your task is to take a user's simple idea and expand it into:
1. An 'Enhanced Positive Prompt': A detailed, descriptive prompt suitable for high-quality image generation. Incorporate vivid details, keywords for quality (e.g., photorealistic, ultra-detailed, cinematic lighting, sharp focus, 8k), specify subject, environment, mood, and style clearly. Aim for around 50-150 words.
2. A 'Negative Prompt': A list of comma-separated terms to *exclude* from the image (e.g., low quality, blurry, text, words, letters, signature, watermark, deformed, ugly, mutated, disfigured, extra limbs, missing limbs, extra fingers, fewer fingers, bad anatomy, unrealistic).

Format your response *exactly* like this, with no additional explanation or conversation before or after:

Enhanced Positive Prompt:
[Your detailed positive prompt here]

Negative Prompt:
[Your comma-separated negative prompt here]"""

        messages = [
            {"role": "user", "content": f"Please enhance this idea for image generation: \"{user_prompt}\""}
        ]

        # Minimal payload based on what worked
        payload = {
            "model": model_id,
            "messages": messages,
            "system": system_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }

        # --- Execute API Call and Process Response ---
        positive_prompt_result = "Error: API Call Failed"
        negative_prompt_result = "Error: API Call Failed"
        print(f"ClaudePromptEnhancer: Sending request to {model_id}...") # Log activity

        try:
            response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=120)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            response_data = response.json()
            print("ClaudePromptEnhancer: Response received.")

            # Extract content
            generated_content = response_data['choices'][0]['message']['content']

            # Parse content
            positive_match = re.search(r"Enhanced Positive Prompt:\s*(.*?)\s*(Negative Prompt:|$)", generated_content, re.DOTALL | re.IGNORECASE)
            negative_match = re.search(r"Negative Prompt:\s*(.*)", generated_content, re.DOTALL | re.IGNORECASE)

            if positive_match and negative_match:
                positive_prompt_result = positive_match.group(1).strip()
                negative_prompt_result = negative_match.group(1).strip()
                print("ClaudePromptEnhancer: Successfully parsed prompts.")
            else:
                print("WARNING: ClaudePromptEnhancer - Could not parse Claude's response format.")
                print("--- Claude Raw Response Start ---")
                print(generated_content)
                print("--- Claude Raw Response End ---")
                positive_prompt_result = "Error: Parsing Failed (Check Console)"
                # Return raw content in negative prompt field for debugging
                negative_prompt_result = generated_content

        except requests.exceptions.HTTPError as e:
            print(f"ERROR: ClaudePromptEnhancer - HTTP Error {e.response.status_code} calling AIMLAPI.")
            try:
                print(f"AIMLAPI Response: {e.response.text[:500]}...") # Print beginning of error
            except: pass
            positive_prompt_result = f"Error: HTTP {e.response.status_code}"
            negative_prompt_result = f"Error: HTTP {e.response.status_code} (Check Console)"
        except requests.exceptions.Timeout:
            print("ERROR: ClaudePromptEnhancer - API request timed out.")
            positive_prompt_result = "Error: Timeout"
            negative_prompt_result = "Error: Timeout"
        except requests.exceptions.RequestException as e:
            print(f"ERROR: ClaudePromptEnhancer - Network or request error: {e}")
            positive_prompt_result = "Error: Request Failed"
            negative_prompt_result = "Error: Request Failed (Check Console)"
        except (KeyError, IndexError, TypeError) as e:
             print(f"ERROR: ClaudePromptEnhancer - Failed to access data in API response: {e}")
             positive_prompt_result = "Error: Bad Response Format"
             negative_prompt_result = "Error: Bad Response Format (Check Console)"
             # Try to log the bad response if possible
             try: print(f"Problematic Response Data: {response_data}")
             except: pass
        except Exception as e:
            print(f"ERROR: ClaudePromptEnhancer - An unexpected error occurred: {e}")
            positive_prompt_result = "Error: Unexpected Node Error"
            negative_prompt_result = "Error: Unexpected Node Error (Check Console)"

        # Return the results as a tuple matching RETURN_TYPES
        return (positive_prompt_result, negative_prompt_result)


# --- ComfyUI Registration ---

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ClaudePromptEnhancer_AIMLAPI": ClaudePromptEnhancer # Use a specific name to avoid clashes
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ClaudePromptEnhancer_AIMLAPI": "Claude Prompt Enhancer (AIMLAPI)"
}