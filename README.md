# Garment Color Matching System

A computer vision pipeline for comparing garments using:

- Human Parsing Segmentation
- CLIP Semantic Similarity
- DINOv2 Region Matching
- Color Histogram Analysis
- Color Correction

## Features

- Automatic garment segmentation
- Reference garment detection
- CLIP-based visual similarity scoring
- DINOv2 patch-level region comparison
- LAB color histogram comparison
- Visual report generation
- CSV and TXT report export

## Project Structure

```text
main.py
segment.py
color_correction.py
color_compare.py
clip_match.py
dino_match.py
dino_region_compare.py
visual_report.py
report.py
requirements.txt
```

## Requirements

- Python 3.12.3

## Installation

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

## Running

Place images inside:

```text
input/
```

Ensure one image starts with:

```text
_color-ref
```

Example:

```text
input/
├── _color-ref-shirt.jpg
├── shirt_1.jpg
├── shirt_2.jpg
```

Run:

```bash
python main.py
```

## Output

Generated files:

```text
outputs/
├── corrected/
├── visual_reports/
├── report.csv
└── report.txt
```

## Models Used

- CLIP (OpenAI)
- DINOv2 (Meta AI)
- FASHN Human Parser

## Tested Environment

- Python 3.12.3
- PyTorch 2.12.0
- Transformers 5.12.0
