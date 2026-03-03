
"""
Agent 01: Image Validation Agent (OpenCLIP version)

Responsibility:
- Decide whether an uploaded image is LIKELY to contain human skin.
- Reject unrelated images (objects, animals, screenshots, etc.)

This agent is a GATEKEEPER. It does NOT:
- Diagnose any condition
- Make medical claims

Techniques used:
1) HSV-based skin detection (coarse filter)
2) Embedding similarity against precomputed NORMAL skin embeddings
"""


# Imports
from PIL import Image
import numpy as np
import cv2

from embeddings import get_embedding, load_embeddings, compute_similarity


# Load NORMAL skin embeddings ONCE
NORMAL_SKIN_EMB_PATH = "normal_skin_embeddings.npy"
normal_skin_embeddings = load_embeddings(NORMAL_SKIN_EMB_PATH)


def is_likely_skin(image: Image.Image) -> bool:
    """
    This checks whether the given image likely contains human skin.

    Parameters:
    - image (PIL.Image): Uploaded image from the user

    Returns:
    - True  -> Image likely contains skin
    - False -> Image likely NOT skin
    """

    # Convert PIL Image to NumPy array
    img = np.array(image)

    # Convert color space from RGB (PIL) to BGR (OpenCV default)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Convert image from BGR to HSV color space
    # In HSV,
    # H -> Hue(color) 
    # S -> Saturation(color intensity) 
    # V -> Value(brightness)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define a broad HSV range that covers most human skin tones
    lower_skin = np.array([0, 20, 40], dtype=np.uint8)
    upper_skin = np.array([25, 255, 255], dtype=np.uint8)

    # Create a binary mask where skin-like pixels are white,now what does this mean?
    # It means :
    # checks every pixel in the HSV image
    # If the pixel’s HSV values fall inside the given range -> mark it as 255 (white)
    # Otherwise → mark it as 0 (black)
    # This is a binary decision mask, not a color judgment
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Calculate how much of the image looks like skin
    skin_pixel_ratio = np.sum(skin_mask > 0) / skin_mask.size

    # If enough pixels look like skin, accept the image
    return skin_pixel_ratio > 0.27

# Function: Embedding-based Similarity Check
def similar_to_normal_skin(image: Image.Image, threshold: float = 0.75) -> bool:
    """
    Check if the uploaded image is visually similar to normal skin.

    Parameters:
    - image (PIL Image): user uploaded image
    - threshold (float): Cosine similarity threshold for acceptance

    Returns:
    - True  -> image is similar to normal skin
    - False -> image is not similar
    """

    # Convert image to embedding
    user_emb = get_embedding(image)

    # Compute similarity with normal skin embeddings
    sims = compute_similarity(user_emb, normal_skin_embeddings)

    # Accept if any similarity >= threshold
    return np.max(sims) >= threshold


# Function 3: Full validation pipeline
def validate_skin_image(image: Image.Image) -> bool:
    """
    Complete validation pipeline combining:
    1) HSV-based skin check
    2) Embedding similarity

    Parameters:
    - image(PIL Image): user uploaded image

    Returns:
    - True  -> image accepted for further analysis
    - False -> ask user for clearer image
    """

    # Load image
    image = image.convert("RGB")

    # Step 1: HSV skin check
    if not is_likely_skin(image):
        print("Image rejected: HSV skin check failed")
        return False

    # Step 2: Embedding similarity
    if not similar_to_normal_skin(image):
        print("Image rejected: Not similar to normal skin")
        return False

    # Passed all checks
    print("Image accepted by Image Validation Agent")
    return True