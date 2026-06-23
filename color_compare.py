# color_compare.py

import cv2
import numpy as np

from skimage.color import deltaE_ciede2000
from scipy.spatial.distance import cosine


# --------------------------------------------------
# GARMENT EXTRACTION
# --------------------------------------------------

def extract_mask_pixels(image, mask):
    """
    Returns garment pixels only.
    """
    return image[mask > 0]


# --------------------------------------------------
# LAB CONVERSION
# --------------------------------------------------

def rgb_to_lab_pixels(rgb_pixels):

    rgb_pixels = rgb_pixels.reshape(-1, 1, 3)

    lab = cv2.cvtColor(
        rgb_pixels.astype(np.uint8),
        cv2.COLOR_RGB2LAB
    )

    return lab.reshape(-1, 3)


# --------------------------------------------------
# DOMINANT COLOR
# --------------------------------------------------

def dominant_color(image, mask):

    pixels = extract_mask_pixels(image, mask)

    if len(pixels) == 0:
        return None

    pixels = pixels.astype(np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    K = 3

    _, labels, centers = cv2.kmeans(
        pixels,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_PP_CENTERS
    )

    counts = np.bincount(labels.flatten())

    dominant = centers[np.argmax(counts)]

    return dominant.astype(np.uint8)


# --------------------------------------------------
# LAB HISTOGRAM
# --------------------------------------------------

def lab_histogram(image, mask):

    garment = image.copy()

    garment[mask == 0] = 0

    lab = cv2.cvtColor(garment, cv2.COLOR_RGB2LAB)

    hist = cv2.calcHist(
        [lab],
        [0, 1, 2],
        mask.astype(np.uint8),
        [8, 8, 8],
        [0, 256,
         0, 256,
         0, 256]
    )

    hist = cv2.normalize(hist, hist)

    return hist.flatten()


# --------------------------------------------------
# HISTOGRAM SIMILARITY
# --------------------------------------------------

def histogram_similarity(hist1, hist2):

    if np.linalg.norm(hist1) == 0:
        return 0

    if np.linalg.norm(hist2) == 0:
        return 0

    score = 1.0 - cosine(hist1, hist2)

    if np.isnan(score):
        return 0.0

    return float(score)


# --------------------------------------------------
# DOMINANT COLOR DELTA-E
# --------------------------------------------------

def dominant_color_similarity(
        color1,
        color2):

    if color1 is None or color2 is None:
        return 0.0

    c1 = np.uint8([[color1]])
    c2 = np.uint8([[color2]])

    lab1 = cv2.cvtColor(
        c1,
        cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    lab2 = cv2.cvtColor(
        c2,
        cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    de = deltaE_ciede2000(
        lab1,
        lab2
    )[0][0]

    similarity = max(
        0,
        100 - de
    )

    return float(similarity)


# --------------------------------------------------
# NORMALIZE GARMENT
# --------------------------------------------------

def normalize_garment(
        image,
        mask,
        size=512):

    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        return None, None

    x1 = np.min(xs)
    x2 = np.max(xs)

    y1 = np.min(ys)
    y2 = np.max(ys)

    crop_img = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]

    crop_img = cv2.resize(
        crop_img,
        (size, size)
    )

    crop_mask = cv2.resize(
        crop_mask,
        (size, size),
        interpolation=cv2.INTER_NEAREST
    )

    return crop_img, crop_mask


# --------------------------------------------------
# REGION SPLIT
# --------------------------------------------------

def split_regions(
        image,
        mask,
        rows=4,
        cols=4):

    h, w = image.shape[:2]

    cell_h = h // rows
    cell_w = w // cols

    regions = []

    for r in range(rows):

        for c in range(cols):

            y1 = r * cell_h
            y2 = (r + 1) * cell_h

            x1 = c * cell_w
            x2 = (c + 1) * cell_w

            region_img = image[y1:y2, x1:x2]

            region_mask = mask[y1:y2, x1:x2]

            regions.append(
                (
                    region_img,
                    region_mask
                )
            )

    return regions


# --------------------------------------------------
# REGION COMPARISON
# --------------------------------------------------

def compare_regions(
        ref_img,
        ref_mask,
        cand_img,
        cand_mask):

    ref_img, ref_mask = normalize_garment(
        ref_img,
        ref_mask
    )

    cand_img, cand_mask = normalize_garment(
        cand_img,
        cand_mask
    )

    if ref_img is None:
        return []

    if cand_img is None:
        return []

    ref_regions = split_regions(
        ref_img,
        ref_mask
    )

    cand_regions = split_regions(
        cand_img,
        cand_mask
    )

    results = []

    for idx in range(len(ref_regions)):

        r_img, r_mask = ref_regions[idx]
        c_img, c_mask = cand_regions[idx]

        h1 = lab_histogram(
            r_img,
            r_mask
        )

        h2 = lab_histogram(
            c_img,
            c_mask
        )

        score = histogram_similarity(
            h1,
            h2
        )

        results.append(
            {
                "region": idx,
                "similarity":
                    round(score * 100, 2)
            }
        )

    return results


# --------------------------------------------------
# OVERALL COLOR SCORE
# --------------------------------------------------

def overall_color_score(
        ref_img,
        ref_mask,
        cand_img,
        cand_mask):

    ref_hist = lab_histogram(
        ref_img,
        ref_mask
    )

    cand_hist = lab_histogram(
        cand_img,
        cand_mask
    )

    hist_score = histogram_similarity(
        ref_hist,
        cand_hist
    )

    ref_dom = dominant_color(
        ref_img,
        ref_mask
    )

    cand_dom = dominant_color(
        cand_img,
        cand_mask
    )

    dom_score = dominant_color_similarity(
        ref_dom,
        cand_dom
    )

    final_score = (
        hist_score * 100 * 0.7 +
        dom_score * 0.3
    )

    return {
        "histogram_similarity":
            round(hist_score * 100, 2),

        "dominant_similarity":
            round(dom_score, 2),

        "overall_similarity":
            round(final_score, 2)
    }