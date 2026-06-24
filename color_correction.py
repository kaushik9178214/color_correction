import cv2
import numpy as np

from sklearn.neighbors import NearestNeighbors


def reinhard_color_transfer(
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

    ref_mean = np.mean(
        ref_pixels,
        axis=0
    )

    ref_std = np.std(
        ref_pixels,
        axis=0
    )

    cand_mean = np.mean(
        cand_pixels,
        axis=0
    )

    cand_std = np.std(
        cand_pixels,
        axis=0
    )

    ref_std = np.maximum(
        ref_std,
        1e-6
    )

    cand_std = np.maximum(
        cand_std,
        1e-6
    )

    corrected_lab = cand_lab.copy()

    for channel in [1,2]:

        corrected_lab[..., channel] = (
            (
                corrected_lab[..., channel]
                - cand_mean[channel]
            )
            *
            (
                ref_std[channel]
                /
                cand_std[channel]
            )
            +
            ref_mean[channel]
        )

    corrected_lab = np.clip(
        corrected_lab,
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

def palette_transfer_color(
        ref_img,
        ref_mask,
        cand_img,
        cand_mask):

    ref_pixels = (
        ref_img[
            ref_mask > 0
        ]
    )

    cand_pixels = (
        cand_img[
            cand_mask > 0
        ]
    )

    if len(ref_pixels) == 0:
        return cand_img

    if len(cand_pixels) == 0:
        return cand_img

    # --------------------------
    # RGB -> LAB
    # --------------------------

    ref_lab_original = cv2.cvtColor(
        ref_pixels.reshape(
            -1,
            1,
            3
        ),
        cv2.COLOR_RGB2LAB
    ).reshape(
        -1,
        3
    ).astype(
        np.float32
    )

    # --------------------------
    # Quantize LAB colors
    # --------------------------

    LAB_STEP = 4

    ref_lab_quantized = (
        np.round(
            ref_lab_original / LAB_STEP
        ) * LAB_STEP
    ).astype(
        np.float32
    )

    # --------------------------
    # Keep unique LAB colors
    # --------------------------

    _, unique_idx = np.unique(
        ref_lab_quantized,
        axis=0,
        return_index=True
    )

    ref_lab = ref_lab_quantized[
        unique_idx
    ]

    ref_pixels = ref_pixels[
        unique_idx
    ]

    print(
        f"Unique LAB colors: "
        f"{len(ref_lab)}"
    )

    cand_lab = cv2.cvtColor(
        cand_pixels.reshape(
            -1,
            1,
            3
        ),
        cv2.COLOR_RGB2LAB
    ).reshape(
        -1,
        3
    ).astype(
        np.float32
    )

    # --------------------------
    # Nearest reference color
    # --------------------------

    nn = NearestNeighbors(
        n_neighbors=1
    )

    nn.fit(
        ref_lab
    )

    _, nearest = nn.kneighbors(
        cand_lab
    )

    nearest = nearest.flatten()

    corrected_pixels = (
        ref_pixels[
            nearest
        ]
    )

    # --------------------------
    # Put back into image
    # --------------------------

    result = cand_img.copy()

    result[
        cand_mask > 0
    ] = corrected_pixels

    return result

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