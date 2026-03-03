"""
medgemma_model.py

Purpose:
Handles loading and inference of the MedGemma multimodal model.

Design Principles:
- Lazy model loading (initializes once on first request)
- GPU acceleration via Hugging Face Spaces decorator
- Safe output extraction
- Graceful failure handling

This module isolates model inference from application logic,
improving modularity and maintainability.
"""

import os
import torch
import spaces
from transformers import pipeline
from PIL import Image


# Hugging Face token loaded securely from environment variables
HF_TOKEN = os.environ.get("hf_token_forjamai")

# MedGemma multimodal instruction-tuned model
MODEL_NAME = "google/medgemma-1.5-4b-it"

# Global pipeline instance (lazy initialization)
pipe = None


@spaces.GPU(duration=120)
def provide_guidance(prompt: str, image: Image.Image = None) -> str:
    """
    Generates multimodal skincare guidance using MedGemma.

    Parameters:
    prompt : str
        Instruction-formatted prompt containing user history.
    image : PIL.Image
        Uploaded skin image.

    Returns:
    str
        Generated text output from the model.

    Notes:
    - Model loads only once (cached globally).
    - GPU allocation is automatically handled by Spaces.
    - Function includes safe extraction and fallback error handling.
    """

    global pipe

    # Lazy load model to avoid repeated initialization
    if pipe is None:
        pipe = pipeline(
            "image-text-to-text",
            model=MODEL_NAME,
            device_map="auto",           # Automatically assign GPU if available
            torch_dtype=torch.bfloat16,  # Efficient memory usage
            token=HF_TOKEN
        )

    # Ensure prompt is clean and non-empty
    clean_prompt = prompt.strip() if prompt else "Analyze this medical image."

    try:
        # Run multimodal inference (image + text)
        outputs = pipe(
                      image=image,
                      text=clean_prompt,
                      max_new_tokens=300,
                      return_full_text=False
                    )

        # Safe structured extraction
        if isinstance(outputs, list) and len(outputs) > 0:
            result = outputs[0]
            if isinstance(result, dict):
                return result.get("generated_text", "")
            return str(result)

        if isinstance(outputs, dict):
            return outputs.get("generated_text", "")

        return str(outputs)

    except Exception as e:
        # Graceful failure mode to avoid UI crash
        return f"Inference Error: {str(e)}"