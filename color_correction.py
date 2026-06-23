import cv2
import numpy as np


def correct_garment_color(
        ref_img,
        ref_mask,
        cand_img,
        cand_mask):

    ref_lab = cv2.cvtColor(
        ref_img,
        cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    cand_lab = cv2.cvtColor(
        cand_img,
        cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    ref_pixels = ref_lab[
        ref_mask > 0
    ]

    cand_pixels = cand_lab[
        cand_mask > 0
    ]

    if len(ref_pixels) == 0:
        return cand_img

    if len(cand_pixels) == 0:
        return cand_img

    ref_a_mean = np.mean(
        ref_pixels[:, 1]
    )

    ref_b_mean = np.mean(
        ref_pixels[:, 2]
    )

    cand_a_mean = np.mean(
        cand_pixels[:, 1]
    )

    cand_b_mean = np.mean(
        cand_pixels[:, 2]
    )

    a_shift = (
        ref_a_mean -
        cand_a_mean
    )

    b_shift = (
        ref_b_mean -
        cand_b_mean
    )

    corrected_lab = cand_lab.copy()

    corrected_lab[..., 1] = np.clip(
        corrected_lab[..., 1] + a_shift,
        0,
        255
    )

    corrected_lab[..., 2] = np.clip(
        corrected_lab[..., 2] + b_shift,
        0,
        255
    )

    corrected_rgb = cv2.cvtColor(
        corrected_lab.astype(np.uint8),
        cv2.COLOR_LAB2RGB
    )

    result = cand_img.copy()

    result[
        cand_mask > 0
    ] = corrected_rgb[
        cand_mask > 0
    ]

    return result