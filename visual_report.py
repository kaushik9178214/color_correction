# visual_report.py

import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_mask(image, mask):

    result = image.copy()

    result[mask == 0] = 0

    return result


def save_visual_report(
        output_path,
        ref_img,
        ref_mask,
        cand_img,
        cand_mask,
        clip_score,
        color_score,
        final_score):

    ref_crop = apply_mask(
        ref_img,
        ref_mask
    )

    cand_crop = apply_mask(
        cand_img,
        cand_mask
    )

    ref_pixels = np.sum(
        ref_mask > 0
    )

    cand_pixels = np.sum(
        cand_mask > 0
    )

    coverage_ratio = (
        min(ref_pixels, cand_pixels)
        /
        max(ref_pixels, cand_pixels)
    ) * 100

    fig = plt.figure(
        figsize=(18, 10)
    )

    # --------------------------------
    # TOP ROW
    # --------------------------------

    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(ref_img)
    ax1.set_title("Reference Image")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(
        ref_mask,
        cmap="gray"
    )
    ax2.set_title("Reference Mask")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 4, 3)
    ax3.imshow(ref_crop)
    ax3.set_title("Reference Garment")
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 4, 4)

    ax4.axis("off")

    ax4.text(
        0.0,
        1.0,
        (
            f"REFERENCE\n\n"
            f"Garment Pixels : {ref_pixels:,}"
        ),
        fontsize=12,
        va="top"
    )

    # --------------------------------
    # BOTTOM ROW
    # --------------------------------

    ax5 = fig.add_subplot(2, 4, 5)
    ax5.imshow(cand_img)
    ax5.set_title("Candidate Image")
    ax5.axis("off")

    ax6 = fig.add_subplot(2, 4, 6)
    ax6.imshow(
        cand_mask,
        cmap="gray"
    )
    ax6.set_title("Candidate Mask")
    ax6.axis("off")

    ax7 = fig.add_subplot(2, 4, 7)
    ax7.imshow(cand_crop)
    ax7.set_title("Candidate Garment")
    ax7.axis("off")

    ax8 = fig.add_subplot(2, 4, 8)

    ax8.axis("off")

    result_text = (
        f"RESULTS\n\n"
        f"CLIP Score : {clip_score:.2f}%\n\n"
        f"Color Score : {color_score:.2f}%\n\n"
        f"Final Score : {final_score:.2f}%\n\n"
        f"Candidate Pixels : {cand_pixels:,}\n\n"
        f"Coverage Ratio : {coverage_ratio:.2f}%"
    )

    if final_score >= 90:
        classification = "VERY STRONG MATCH"

    elif final_score >= 80:
        classification = "STRONG MATCH"

    elif final_score >= 70:
        classification = "GOOD MATCH"

    elif final_score >= 60:
        classification = "POSSIBLE MATCH"

    else:
        classification = "WEAK MATCH"

    result_text += (
        f"\n\nClassification:\n"
        f"{classification}"
    )

    ax8.text(
        0.0,
        1.0,
        result_text,
        fontsize=12,
        va="top"
    )

    plt.suptitle(
        "Garment Matching Report",
        fontsize=18
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        bbox_inches="tight"
    )

    plt.close()