"""
Agent 02: Image Similarity Agent (OpenCLIP version)

Responsibility:
- Compare user uploaded image to skin condition reference images
- Compute similarity score (LOW / MEDIUM / HIGH)
- Does NOT diagnose or give treatment

Techniques used:
1) Embed user image using OpenCLIP
2) Cosine similarity against precomputed skin condition embeddings
"""


# Imports
from  PIL import Image
from embeddings import get_embedding, load_embeddings, compute_similarity
import numpy as np


# Load skin condition embeddings ONCE
CONDITION_EMB_PATH = "skin_condition_embeddings.npy"
condition_embeddings = load_embeddings(CONDITION_EMB_PATH)

def similarity_level(image: Image.Image) -> str:
    """
    Compare a user image embedding to all skin condition embeddings
    and return only the similarity category: LOW / MEDIUM / HIGH.

    Parameters:
    - image (PIL.Image): User uploaded image

    Returns:
    - str: "LOW", "MEDIUM", or "HIGH"
    """

    # Step 1: Embed user image
    user_emb = get_embedding(image)

    # Step 2: Compute cosine similarity to all condition embeddings
    sims = compute_similarity(user_emb, condition_embeddings)

    # Step 3: Get the top similarity
    top_score = np.max(sims)

    # Step 4: Map score to category
    if top_score < 0.55:
        return "LOW"
    elif top_score < 0.80:
        return "MEDIUM"
    else:
        return "HIGH"
