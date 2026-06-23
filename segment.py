# segment.py

from fashn_human_parser import FashnHumanParser

parser = FashnHumanParser()

import cv2
import numpy as np


# --------------------------------------------------
# IMAGE LOADING
# --------------------------------------------------

# changes
REFERENCE_LABEL = None


def set_reference_labels(image):

    global REFERENCE_LABEL

    parsing = parser.predict(image)

    clothing_mask = np.isin(
        parsing,
        [3, 4, 5, 6]
    ).astype(np.uint8)

    clothing_mask = clean_mask(
        clothing_mask
    )

    clothing_mask = largest_component(
        clothing_mask
    )

    labels_inside_mask = parsing[
        clothing_mask > 0
    ]

    unique, counts = np.unique(
        labels_inside_mask,
        return_counts=True
    )

    valid_mask = unique != 0

    unique = unique[valid_mask]
    counts = counts[valid_mask]

    if len(unique) == 0:

        raise Exception(
            "No garment label found in reference image."
        )

    REFERENCE_LABEL = int(
        unique[
            np.argmax(counts)
        ]
    )

    print(
        f"Reference label: "
        f"{REFERENCE_LABEL} "
        f"({parser.get_label_name(REFERENCE_LABEL)})"
    )

def load_image(path):

    image = cv2.imread(path)

    if image is None:
        raise Exception(
            f"Unable to load image: {path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image


# --------------------------------------------------
# SIMPLE FOREGROUND EXTRACTION
# --------------------------------------------------

def grabcut_foreground(image):

    mask = np.zeros(
        image.shape[:2],
        np.uint8
    )

    h, w = image.shape[:2]

    rect = (
        int(w * 0.05),
        int(h * 0.05),
        int(w * 0.90),
        int(h * 0.90)
    )

    bgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    fgd_model = np.zeros(
        (1, 65),
        np.float64
    )

    cv2.grabCut(
        image,
        mask,
        rect,
        bgd_model,
        fgd_model,
        5,
        cv2.GC_INIT_WITH_RECT
    )

    mask = np.where(
        (mask == 2) |
        (mask == 0),
        0,
        1
    ).astype("uint8")

    return mask


# --------------------------------------------------
# MORPH CLEANUP
# --------------------------------------------------

def clean_mask(mask):

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


# --------------------------------------------------
# KEEP LARGEST COMPONENT
# --------------------------------------------------

def largest_component(mask):

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    if num_labels <= 1:
        return mask

    largest_label = 1

    largest_area = (
        stats[1, cv2.CC_STAT_AREA]
    )

    for i in range(2, num_labels):

        area = (
            stats[i, cv2.CC_STAT_AREA]
        )

        if area > largest_area:

            largest_area = area
            largest_label = i

    result = np.zeros_like(mask)

    result[
        labels == largest_label
    ] = 1

    return result


# --------------------------------------------------
# REFERENCE IMAGE DETECTION
# --------------------------------------------------

def is_reference_image(filename):

    return (
        filename.lower()
        .startswith("_color-ref")
    )


# --------------------------------------------------
# GARMENT MASK GENERATION
# --------------------------------------------------

def segment_image(image):

    global REFERENCE_LABEL

    parsing = parser.predict(image)

    print(
        "Labels:",
        np.unique(parsing)
    )

    if REFERENCE_LABEL is None:

        clothing_mask = np.isin(
            parsing,
            [3,4,5,6]
        ).astype(np.uint8)

        clothing_mask = clean_mask(
            clothing_mask
        )

        clothing_mask = largest_component(
            clothing_mask
        )

        return clothing_mask


    print(
        f"Using reference label: {REFERENCE_LABEL}"
    )

    mask = (
        parsing == REFERENCE_LABEL
    ).astype(np.uint8)

    mask = clean_mask(
        mask
    )

    mask = largest_component(
        mask
    )

    return mask


# --------------------------------------------------
# CROPPING
# --------------------------------------------------

def crop_using_mask(
        image,
        mask):

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:

        return image

    x1 = np.min(xs)
    x2 = np.max(xs)

    y1 = np.min(ys)
    y2 = np.max(ys)

    crop = image[
        y1:y2,
        x1:x2
    ]

    return crop


# --------------------------------------------------
# COVERAGE
# --------------------------------------------------

def mask_coverage(mask):

    garment_pixels = np.sum(
        mask > 0
    )

    total_pixels = (
        mask.shape[0] *
        mask.shape[1]
    )

    return (
        garment_pixels /
        total_pixels
    ) * 100.0