import cv2
import numpy as np

from scipy.spatial.distance import cdist

from clip_match import garment_crop
from dino_match import extract_patch_features


# --------------------------------------------------
# PATCH COLORS
# --------------------------------------------------

def patch_lab_colors(
        image,
        grid_size=16):

    image = cv2.resize(
        image,
        (224, 224)
    )

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2LAB
    )

    patch_w = 224 // grid_size
    patch_h = 224 // grid_size

    colors = []

    for row in range(grid_size):

        for col in range(grid_size):

            y1 = row * patch_h
            y2 = (row + 1) * patch_h

            x1 = col * patch_w
            x2 = (col + 1) * patch_w

            patch = lab[
                y1:y2,
                x1:x2
            ]

            colors.append(
                patch
                .reshape(-1, 3)
                .mean(axis=0)
            )

    return np.array(colors)


# --------------------------------------------------
# MUTUAL MATCHES
# --------------------------------------------------

def mutual_matches(
        ref_features,
        cand_features):

    sim = (
        1 -
        cdist(
            ref_features,
            cand_features,
            metric="cosine"
        )
    )

    ref_to_cand = np.argmax(
        sim,
        axis=1
    )

    cand_to_ref = np.argmax(
        sim,
        axis=0
    )

    matches = []

    for ref_idx, cand_idx in enumerate(
            ref_to_cand):

        if (
            cand_to_ref[cand_idx]
            == ref_idx
        ):

            matches.append(
                (
                    ref_idx,
                    cand_idx,
                    sim[
                        ref_idx,
                        cand_idx
                    ]
                )
            )

    return matches


# --------------------------------------------------
# LAB COLOR SIMILARITY
# --------------------------------------------------

def color_similarity(
        lab1,
        lab2):

    dist = np.linalg.norm(
        lab1 - lab2
    )

    score = max(
        0,
        100 - dist
    )

    return float(score)


# --------------------------------------------------
# REGION COMPARISON
# --------------------------------------------------

def compare_regions(
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
        return []

    if cand_crop is None:
        return []

    ref_crop = cv2.resize(
        ref_crop,
        (224, 224)
    )

    cand_crop = cv2.resize(
        cand_crop,
        (224, 224)
    )

    ref_features = (
        extract_patch_features(
            ref_crop
        )
    )

    cand_features = (
        extract_patch_features(
            cand_crop
        )
    )

    matches = mutual_matches(
        ref_features,
        cand_features
    )

    ref_colors = patch_lab_colors(
        ref_crop
    )

    cand_colors = patch_lab_colors(
        cand_crop
    )

    results = []

    for idx, (
            ref_patch,
            cand_patch,
            dino_score
    ) in enumerate(matches):

        score = color_similarity(
            ref_colors[
                ref_patch
            ],
            cand_colors[
                cand_patch
            ]
        )

        results.append(
            {
                "region": idx,
                "similarity":
                    round(
                        score,
                        2
                    ),

                "ref_patch":
                    int(
                        ref_patch
                    ),

                "cand_patch":
                    int(
                        cand_patch
                    ),

                "dino_score":
                    round(
                        float(
                            dino_score
                        ) * 100,
                        2
                    )
            }
        )

    return results