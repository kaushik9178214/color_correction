import torch
import numpy as np

from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoModel
)

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Loading DINOv2...")

MODEL_NAME = "facebook/dinov2-small"

processor = (
    AutoImageProcessor
    .from_pretrained(MODEL_NAME)
)

model = (
    AutoModel
    .from_pretrained(MODEL_NAME)
)

model = model.to(DEVICE)
model.eval()

print("DINOv2 loaded.")


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


@torch.no_grad()
def extract_patch_features(
        image):

    pil = Image.fromarray(image)

    inputs = processor(
        images=pil,
        return_tensors="pt"
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    outputs = model(
        **inputs
    )

    features = (
        outputs
        .last_hidden_state
    )

    patch_tokens = (
        features[:, 1:, :]
    )

    patch_tokens = (
        patch_tokens
        .squeeze(0)
    )

    patch_tokens = (
        patch_tokens /
        patch_tokens.norm(
            dim=1,
            keepdim=True
        )
    )

    return (
        patch_tokens
        .cpu()
        .numpy()
    )