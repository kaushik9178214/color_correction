import cv2
import numpy as np
import matplotlib.pyplot as plt

from segment import (
    load_image,
    segment_image,
    set_reference_labels
)

from dino_match import (
    garment_crop,
    extract_patch_features
)

from scipy.spatial.distance import cdist


# ----------------------------------------
# CONFIG
# ----------------------------------------

REF = "input/_color-ref-1003229806_15000.jpeg"

CAND = "input/100322980615000_2.jpeg"

OUTPUT = "outputs/dino_matches.png"

TOP_K = 50


# ----------------------------------------
# PATCH CENTERS
# ----------------------------------------

def patch_centers(
        width,
        height,
        grid_size=16):

    centers = []

    patch_w = width / grid_size
    patch_h = height / grid_size

    for row in range(grid_size):

        for col in range(grid_size):

            x = (
                col * patch_w
                + patch_w / 2
            )

            y = (
                row * patch_h
                + patch_h / 2
            )

            centers.append(
                (x, y)
            )

    return np.array(
        centers
    )


# ----------------------------------------
# MUTUAL MATCHES
# ----------------------------------------

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

    matches.sort(
        key=lambda x: x[2],
        reverse=True
    )

    return matches


# ----------------------------------------
# MAIN
# ----------------------------------------

print(
    "\nLoading images..."
)

ref_img = load_image(
    REF
)

set_reference_labels(
    ref_img
)

ref_mask = segment_image(
    ref_img
)

cand_img = load_image(
    CAND
)

cand_mask = segment_image(
    cand_img
)

print(
    "Cropping garments..."
)

ref_crop = garment_crop(
    ref_img,
    ref_mask
)

cand_crop = garment_crop(
    cand_img,
    cand_mask
)

print(
    "Extracting DINO features..."
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

print(
    "Finding mutual matches..."
)

matches = mutual_matches(
    ref_features,
    cand_features
)

print(
    f"Mutual matches: "
    f"{len(matches)}"
)

matches = matches[
    :TOP_K
]

# ----------------------------------------
# RESIZE TO DINO SIZE
# ----------------------------------------

ref_vis = cv2.resize(
    ref_crop,
    (224, 224)
)

cand_vis = cv2.resize(
    cand_crop,
    (224, 224)
)

# ----------------------------------------
# CANVAS
# ----------------------------------------

canvas = np.ones(
    (
        224,
        224 * 2,
        3
    ),
    dtype=np.uint8
) * 255

canvas[
    :224,
    :224
] = ref_vis

canvas[
    :224,
    224:
] = cand_vis

# ----------------------------------------
# PATCH CENTERS
# ----------------------------------------

ref_centers = patch_centers(
    224,
    224
)

cand_centers = patch_centers(
    224,
    224
)

# ----------------------------------------
# DRAW MATCHES
# ----------------------------------------

for ref_idx, cand_idx, score in matches:

    x1, y1 = ref_centers[
        ref_idx
    ]

    x2, y2 = cand_centers[
        cand_idx
    ]

    x2 += 224

    p1 = (
        int(x1),
        int(y1)
    )

    p2 = (
        int(x2),
        int(y2)
    )

    cv2.line(
        canvas,
        p1,
        p2,
        (0,255,0),
        1
    )

    cv2.circle(
        canvas,
        p1,
        2,
        (255,0,0),
        -1
    )

    cv2.circle(
        canvas,
        p2,
        2,
        (255,0,0),
        -1
    )

# ----------------------------------------
# SAVE
# ----------------------------------------

plt.figure(
    figsize=(14,7)
)

plt.imshow(
    canvas
)

plt.axis(
    "off"
)

plt.tight_layout()

plt.savefig(
    OUTPUT,
    bbox_inches="tight"
)

plt.close()

print(
    f"\nSaved: {OUTPUT}"
)