# test_mask.py

import os
import cv2
import numpy as np

from segment import (
    load_image,
    segment_image
)

os.makedirs(
    "debug_masks",
    exist_ok=True
)

for file in os.listdir("input"):

    path = os.path.join(
        "input",
        file
    )

    image = load_image(path)

    mask = segment_image(
        image
    )

    cv2.imwrite(
        os.path.join(
            "debug_masks",
            file + "_mask.png"
        ),
        mask * 255
    )

    print(
        file,
        np.unique(mask)
    )