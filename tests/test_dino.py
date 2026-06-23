from segment import (
    load_image,
    segment_image,
    set_reference_labels
)

from dino_match import (
    garment_crop,
    extract_patch_features
)

from dino_correspondence import (
    dense_match,
    overall_match_score
)

REF = (
    "input/_color-ref-1003229806_15000.jpeg"
)

CAND = (
    "input/100322980615000_2.jpeg"
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

matches, scores = (
    dense_match(
        ref_features,
        cand_features
    )
)

print(
    "Patch count:",
    len(scores)
)

print(
    "DINO score:",
    round(
        overall_match_score(
            scores
        ),
        2
    )
)