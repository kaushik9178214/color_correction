# main.py

import os
import glob
import cv2

from segment import (
    load_image,
    segment_image,
    set_reference_labels,
    is_reference_image
)

from color_correction import (
    correct_garment_color
)

from clip_match import (
    compare_garments
)

from dino_region_compare import (
    compare_regions
)

from color_compare import (
    overall_color_score
)

from visual_report import (
    save_visual_report
)

from report import (
    write_csv_report,
    write_txt_report,
    calculate_final_score,
    print_summary,
    ensure_output_dir
)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

INPUT_DIR = "input"

OUTPUT_DIR = "outputs"

VISUAL_DIR = os.path.join(
    OUTPUT_DIR,
    "visual_reports"
)

CORRECTED_DIR = os.path.join(
    OUTPUT_DIR,
    "corrected"
)

ensure_output_dir(
    VISUAL_DIR
)

ensure_output_dir(
    CORRECTED_DIR
)

CSV_REPORT = os.path.join(
    OUTPUT_DIR,
    "report.csv"
)

TXT_REPORT = os.path.join(
    OUTPUT_DIR,
    "report.txt"
)


# --------------------------------------------------
# FIND REFERENCE
# --------------------------------------------------

def find_reference(files):

    for file in files:

        name = os.path.basename(file)

        if is_reference_image(name):
            return file

    raise Exception(
        "No _color-ref image found."
    )


# --------------------------------------------------
# IMAGE LIST
# --------------------------------------------------

def load_image_paths(folder):

    exts = (
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.webp"
    )

    files = []

    for ext in exts:

        files.extend(
            glob.glob(
                os.path.join(
                    folder,
                    ext
                )
            )
        )

    return sorted(files)


# --------------------------------------------------
# PROCESS
# --------------------------------------------------

def process_folder():

    ensure_output_dir(
        OUTPUT_DIR
    )

    files = load_image_paths(
        INPUT_DIR
    )

    if len(files) == 0:

        raise Exception(
            "No images found."
        )

    reference_path = find_reference(
        files
    )

    print()
    print(
        f"Reference : "
        f"{os.path.basename(reference_path)}"
    )

    # --------------------------------

    ref_img = load_image(
        reference_path
    )

    set_reference_labels(ref_img)

    ref_mask = segment_image(
        ref_img
    )

    results = []

    # --------------------------------

    for file in files:

        if file == reference_path:
            continue

        print()

        print(
            f"Processing : "
            f"{os.path.basename(file)}"
        )

        cand_img = load_image(
            file
        )

        cand_mask = segment_image(
            cand_img
        )

        corrected_img = correct_garment_color(
            ref_img,
            ref_mask,
            cand_img,
            cand_mask
        )

        corrected_path = os.path.join(
            CORRECTED_DIR,
            os.path.basename(file)
        )

        cv2.imwrite(
            corrected_path,
            cv2.cvtColor(
                corrected_img,
                cv2.COLOR_RGB2BGR
            )
        )

        # -----------------------------
        # COLOR CORRECTION
        # -----------------------------

        # -----------------------------
        # CLIP
        # -----------------------------

        clip_score = compare_garments(
            ref_img,
            ref_mask,
            cand_img,
            cand_mask
        )

        # -----------------------------
        # COLOR
        # -----------------------------

        color_result = overall_color_score(
            ref_img,
            ref_mask,
            cand_img,
            cand_mask
        )

        # -----------------------------
        # REGIONS
        # -----------------------------

        region_result = compare_regions(
            ref_img,
            ref_mask,
            cand_img,
            cand_mask
        )

        # -----------------------------
        # FINAL SCORE
        # -----------------------------

        final_score = calculate_final_score(
            clip_score,
            color_result
        )

        report_path = os.path.join(
            VISUAL_DIR,
            os.path.splitext(
                os.path.basename(file)
            )[0] + ".png"
        )

        save_visual_report(
            report_path,
            ref_img,
            ref_mask,
            cand_img,
            cand_mask,
            clip_score,
            color_result[
                "overall_similarity"
            ],
            final_score
        )

        result = {

            "image":
                os.path.basename(file),

            "clip_similarity":
                clip_score,

            "color":
                color_result,

            "regions":
                region_result,

            "final_score":
                final_score
        }

        results.append(
            result
        )

    # --------------------------------
    # SORT
    # --------------------------------

    results.sort(
        key=lambda x:
        x["final_score"],
        reverse=True
    )

    # --------------------------------
    # REPORTS
    # --------------------------------

    write_csv_report(
        CSV_REPORT,
        results
    )

    write_txt_report(
        TXT_REPORT,
        os.path.basename(
            reference_path
        ),
        results
    )

    print_summary(
        results
    )

    print()

    print(
        f"CSV Report : "
        f"{CSV_REPORT}"
    )

    print(
        f"TXT Report : "
        f"{TXT_REPORT}"
    )

    print()


# --------------------------------------------------
# ENTRY
# --------------------------------------------------

if __name__ == "__main__":

    process_folder()