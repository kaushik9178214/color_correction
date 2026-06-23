# report.py

import os
import csv
import numpy as np

from datetime import datetime


# --------------------------------------------------
# CREATE OUTPUT DIRECTORY
# --------------------------------------------------

def ensure_output_dir(path):

    os.makedirs(
        path,
        exist_ok=True
    )


# --------------------------------------------------
# CSV REPORT
# --------------------------------------------------

def write_csv_report(
        output_file,
        results):

    fields = [
        "image",
        "clip_similarity",
        "histogram_similarity",
        "dominant_similarity",
        "overall_similarity",
        "avg_patch_similarity",
        "best_patch_similarity",
        "worst_patch_similarity",
        "total_patch_matches"
    ]

    with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8") as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for row in results:

            region_scores = [
                r["similarity"]
                for r in row["regions"]
            ]

            if len(region_scores):

                avg_patch = round(
                    np.mean(region_scores),
                    2
                )

                best_patch = round(
                    np.max(region_scores),
                    2
                )

                worst_patch = round(
                    np.min(region_scores),
                    2
                )

            else:

                avg_patch = 0
                best_patch = 0
                worst_patch = 0

            writer.writerow(
                {
                    "image":
                        row["image"],

                    "clip_similarity":
                        row["clip_similarity"],

                    "histogram_similarity":
                        row["color"][
                            "histogram_similarity"
                        ],

                    "dominant_similarity":
                        row["color"][
                            "dominant_similarity"
                        ],

                    "overall_similarity":
                        row["final_score"],

                    "avg_patch_similarity":
                        avg_patch,

                    "best_patch_similarity":
                        best_patch,

                    "worst_patch_similarity":
                        worst_patch,

                    "total_patch_matches":
                        len(
                            row["regions"]
                        )
                }
            )


# --------------------------------------------------
# MATCH LABEL
# --------------------------------------------------

def classify_score(score):

    if score >= 90:
        return "VERY STRONG MATCH"

    if score >= 80:
        return "STRONG MATCH"

    if score >= 70:
        return "GOOD MATCH"

    if score >= 60:
        return "POSSIBLE MATCH"

    return "WEAK MATCH"


# --------------------------------------------------
# REGION ANALYSIS
# --------------------------------------------------

def build_region_text(
        region_scores):

    if len(region_scores) == 0:
        return "No region matches found."

    similarities = [
        r["similarity"]
        for r in region_scores
    ]

    avg_score = np.mean(
        similarities
    )

    best = max(
        region_scores,
        key=lambda x:
        x["similarity"]
    )

    worst = min(
        region_scores,
        key=lambda x:
        x["similarity"]
    )

    strong = len([
        x for x in region_scores
        if x["similarity"] >= 90
    ])

    medium = len([
        x for x in region_scores
        if 80 <= x["similarity"] < 90
    ])

    weak = len([
        x for x in region_scores
        if x["similarity"] < 80
    ])

    lines = []

    lines.append(
        f"Total Matches : {len(region_scores)}"
    )

    lines.append(
        f"Average Similarity : {avg_score:.2f}%"
    )

    lines.append(
        f"Best Match : {best['similarity']:.2f}%"
    )

    lines.append(
        f"Worst Match : {worst['similarity']:.2f}%"
    )

    lines.append(
        f"Strong Matches (>90%) : {strong}"
    )

    lines.append(
        f"Medium Matches (80-90%) : {medium}"
    )

    lines.append(
        f"Weak Matches (<80%) : {weak}"
    )

    lines.append("")
    lines.append(
        "Patch Correspondence Analysis"
    )

    lines.append(
        "-" * 40
    )

    for region in region_scores:

        line = (
            f"Match "
            f"{region['region']:03d}"
            f" : "
            f"{region['similarity']:.2f}%"
        )

        if "ref_patch" in region:

            line += (
                f" | Ref Patch "
                f"{region['ref_patch']}"
            )

        if "cand_patch" in region:

            line += (
                f" -> Cand Patch "
                f"{region['cand_patch']}"
            )

        if "dino_score" in region:

            line += (
                f" | DINO "
                f"{region['dino_score']:.2f}%"
            )

        lines.append(
            line
        )

    return "\n".join(lines)


# --------------------------------------------------
# TXT REPORT
# --------------------------------------------------

def write_txt_report(
        output_file,
        reference_name,
        results):

    lines = []

    lines.append(
        "=" * 70
    )

    lines.append(
        "GARMENT ANALYSIS REPORT"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")

    lines.append(
        f"Generated: "
        f"{datetime.now()}"
    )

    lines.append(
        f"Reference: "
        f"{reference_name}"
    )

    lines.append("")

    results = sorted(
        results,
        key=lambda x:
        x["final_score"],
        reverse=True
    )

    for rank, item in enumerate(
            results,
            start=1):

        lines.append(
            "-" * 70
        )

        lines.append(
            f"RANK #{rank}"
        )

        lines.append(
            "-" * 70
        )

        lines.append(
            f"Image : "
            f"{item['image']}"
        )

        lines.append("")

        lines.append(
            f"CLIP Similarity : "
            f"{item['clip_similarity']:.2f}%"
        )

        lines.append(
            f"Histogram Similarity : "
            f"{item['color']['histogram_similarity']:.2f}%"
        )

        lines.append(
            f"Dominant Color Similarity : "
            f"{item['color']['dominant_similarity']:.2f}%"
        )

        lines.append(
            f"Final Score : "
            f"{item['final_score']:.2f}%"
        )

        lines.append(
            f"Classification : "
            f"{classify_score(item['final_score'])}"
        )

        region_scores = [
            r["similarity"]
            for r in item["regions"]
        ]

        if len(region_scores):

            lines.append("")

            lines.append(
                f"Patch Matches : "
                f"{len(region_scores)}"
            )

            lines.append(
                f"Average Patch Similarity : "
                f"{np.mean(region_scores):.2f}%"
            )

            lines.append(
                f"Best Patch Similarity : "
                f"{np.max(region_scores):.2f}%"
            )

            lines.append(
                f"Worst Patch Similarity : "
                f"{np.min(region_scores):.2f}%"
            )

        lines.append("")
        lines.append(
            "Region Analysis"
        )

        lines.append("")

        lines.append(
            build_region_text(
                item["regions"]
            )
        )

        lines.append("")
        lines.append("")

    with open(
            output_file,
            "w",
            encoding="utf-8") as f:

        f.write(
            "\n".join(lines)
        )


# --------------------------------------------------
# FINAL SCORE
# --------------------------------------------------

def calculate_final_score(
        clip_similarity,
        color_result):

    score = (
        clip_similarity * 0.60 +
        color_result[
            "overall_similarity"
        ] * 0.40
    )

    return round(
        score,
        2
    )


# --------------------------------------------------
# CONSOLE SUMMARY
# --------------------------------------------------

def print_summary(results):

    print()

    print(
        "=" * 60
    )

    print(
        "FINAL RANKING"
    )

    print(
        "=" * 60
    )

    results = sorted(
        results,
        key=lambda x:
        x["final_score"],
        reverse=True
    )

    for idx, item in enumerate(
            results,
            start=1):

        print()

        print(
            f"{idx}. "
            f"{item['image']}"
        )

        print(
            f"Score : "
            f"{item['final_score']:.2f}%"
        )

        print(
            f"Result : "
            f"{classify_score(item['final_score'])}"
        )

        region_scores = [
            r["similarity"]
            for r in item["regions"]
        ]

        if len(region_scores):

            print(
                f"Patch Matches : "
                f"{len(region_scores)}"
            )

            print(
                f"Avg Patch Similarity : "
                f"{np.mean(region_scores):.2f}%"
            )

    print()