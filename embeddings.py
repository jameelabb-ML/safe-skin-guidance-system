"""
embeddings.py

Purpose:
Provides embedding and similarity utilities for the Skin AI system.

Responsibilities:
1. Convert images into normalized vector embeddings.
2. Generate reference embeddings (one-time setup).
3. Load stored embeddings.
4. Compute cosine similarity for comparison.

Design Decisions:
- Reference images are embedded once during setup.
- Only .npy embedding files are deployed.
- Raw reference images are NOT shipped with the application.
- This keeps deployment lightweight and privacy-conscious.
"""

import os
import numpy as np
import torch
import open_clip
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity


# Device configuration (CPU for compatibility in lightweight environments)
device = "cpu"

# Initialize model and preprocessing transforms once
# ViT-B-32 chosen for balanced performance and resource efficiency
model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai',
    device=device
)

model.eval()  # Disable dropout & training behavior


def get_embedding(image: Image.Image) -> np.ndarray:
    """
    Converts a PIL image into a normalized embedding vector.

    Steps:
    1. Convert image to RGB.
    2. Apply preprocessing transform.
    3. Run forward pass through model.
    4. Normalize vector (unit length).

    Returns:
    np.ndarray
        Flattened 1D normalized embedding vector.
    """

    image = image.convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image_tensor)

    # Normalize embedding to unit vector
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)

    return embedding.cpu().numpy().flatten()


def embed_reference_images():
    """
    ONE-TIME SETUP FUNCTION.

    Generates embeddings for:
    - Normal skin reference images
    - Skin condition reference images

    Output:
    Saves .npy files containing:
    - Embeddings
    - Corresponding filenames

    NOTE:
    This function is NOT executed in production.
    Only precomputed embeddings are deployed.
    """

    
    # Normal Skin Embeddings
    NORMAL_DIR = "data/reference_images/normal_skin"
    NORMAL_EMBED_FILE = "normal_skin_embeddings.npy"


    normal_embeddings = []
    

    for file in os.listdir(NORMAL_DIR):
        if not file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")):
            continue

        path = os.path.join(NORMAL_DIR, file)

        try:
            image = Image.open(path)
            emb = get_embedding(image)
            normal_embeddings.append(emb)
            
        except Exception as e:
            print(f"Skipping normal image {file}: {e}")

    np.save(NORMAL_EMBED_FILE, np.array(normal_embeddings))
    

    print("Normal skin embeddings saved.")

    
    # Skin Condition Embeddings
    CONDITION_DIR = "data/reference_images/skin_condition"
    CONDITION_EMBED_FILE = "skin_condition_embeddings.npy"
    

    condition_embeddings = []

    for condition in os.listdir(CONDITION_DIR):
        condition_path = os.path.join(CONDITION_DIR, condition)

        if not os.path.isdir(condition_path):
            continue

        for file in os.listdir(condition_path):
            if not file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".webp")):
                continue

            path = os.path.join(condition_path, file)

            try:
                image = Image.open(path)
                emb = get_embedding(image)
                condition_embeddings.append(emb)
            except Exception as e:
                print(f"Skipping condition image {file}: {e}")

    np.save(CONDITION_EMBED_FILE, np.array(condition_embeddings))

    print("Skin condition embeddings saved.")


def load_embeddings(file_path: str) -> np.ndarray:
    """
    Loads precomputed embeddings from disk.

    Parameters:

    file_path : str
        Path to .npy embedding file.

    Returns:
    np.ndarray
        Loaded embedding matrix.
    """
    return np.load(file_path)


def compute_similarity(
    user_embedding: np.ndarray,
    reference_embeddings: np.ndarray
) -> np.ndarray:
    """
    Computes cosine similarity between user embedding
    and reference embedding matrix.

    Higher value → greater similarity.

    Returns:
    np.ndarray
        Array of similarity scores.
    """
    return cosine_similarity([user_embedding], reference_embeddings)[0]