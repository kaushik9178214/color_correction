# clip_match.py

import torch
import numpy as np

from PIL import Image

from transformers import (
    CLIPProcessor,
    CLIPModel
)

# --------------------------------------------------
# DEVICE
# --------------------------------------------------

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading CLIP...")

MODEL_NAME = "openai/clip-vit-base-patch32"

model = CLIPModel.from_pretrained(
    MODEL_NAME
)

processor = CLIPProcessor.from_pretrained(
    MODEL_NAME
)

model = model.to(DEVICE)
model.eval()

print("CLIP loaded.")


# --------------------------------------------------
# MASKED GARMENT EXTRACTION
# --------------------------------------------------

def garment_crop(
        image,
        mask,
        padding=10):

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None

    x1 = max(0, np.min(xs) - padding)
    x2 = min(
        image.shape[1],
        np.max(xs) + padding
    )

    y1 = max(0, np.min(ys) - padding)
    y2 = min(
        image.shape[0],
        np.max(ys) + padding
    )

    crop = image[y1:y2, x1:x2]

    crop_mask = mask[y1:y2, x1:x2]

    crop = crop.copy()

    crop[crop_mask == 0] = 255

    return crop


# --------------------------------------------------
# IMAGE -> EMBEDDING
# --------------------------------------------------

@torch.no_grad()
def image_embedding(image):

    pil = Image.fromarray(image)

    inputs = processor(
        images=pil,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    outputs = model.get_image_features(
        **inputs
    )

    # Transformers version compatibility
    if hasattr(outputs, "pooler_output"):
        features = outputs.pooler_output
    else:
        features = outputs

    features = (
        features
        .detach()
        .cpu()
        .numpy()[0]
    )

    features = (
        features /
        np.linalg.norm(features)
    )

    return features


# --------------------------------------------------
# EMBEDDING SIMILARITY
# --------------------------------------------------

def embedding_similarity(
        emb1,
        emb2):

    score = np.dot(
        emb1,
        emb2
    )

    return float(score)


# --------------------------------------------------
# REFERENCE VS CANDIDATE
# --------------------------------------------------

def compare_garments(
        ref_img,
        ref_mask,
        cand_img,
        cand_mask):

    ref_crop = garment_crop(
        ref_img,
        ref_mask
    )

    cand_crop = garment_crop(
        cand_img,
        cand_mask
    )

    if ref_crop is None:
        return 0.0

    if cand_crop is None:
        return 0.0

    ref_emb = image_embedding(
        ref_crop
    )

    cand_emb = image_embedding(
        cand_crop
    )

    score = embedding_similarity(
        ref_emb,
        cand_emb
    )

    return round(
        score * 100,
        2
    )


# --------------------------------------------------
# MULTIPLE CANDIDATES
# --------------------------------------------------

def rank_candidates(
        ref_img,
        ref_mask,
        candidates):

    """
    candidates:

    [
        {
            "name": "...",
            "image": img,
            "mask": mask
        }
    ]
    """

    rankings = []

    ref_crop = garment_crop(
        ref_img,
        ref_mask
    )

    if ref_crop is None:
        return []

    ref_emb = image_embedding(
        ref_crop
    )

    for item in candidates:

        crop = garment_crop(
            item["image"],
            item["mask"]
        )

        if crop is None:
            continue

        emb = image_embedding(
            crop
        )

        score = embedding_similarity(
            ref_emb,
            emb
        )

        rankings.append(
            {
                "name":
                    item["name"],

                "clip_similarity":
                    round(
                        score * 100,
                        2
                    )
            }
        )

    rankings.sort(
        key=lambda x:
        x["clip_similarity"],
        reverse=True
    )

    return rankings