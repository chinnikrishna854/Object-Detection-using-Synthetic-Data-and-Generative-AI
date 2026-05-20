# Object Detection using Synthetic Data and Generative AI

> Implementation of the paper **"Object Detection using Synthetic Data and Generative AI"**  
> Published at ICSCDS-2025 (IEEE) — VIT-AP University, Amaravati, India

---

## Overview

This project demonstrates that synthetic datasets generated using Generative AI can match or outperform real-world data for training autonomous vehicle perception systems. The full pipeline covers synthetic image generation, automated annotation, YOLOv8 training for detection and segmentation, and BLIP-based image captioning for scenario understanding.

---

## Pipeline

```
Prompt Generation → Image Generation → SAM Annotation → Dataset Prep → YOLOv8 Training → Evaluation → BLIP Captioning
```

| Step | Phase | Tool / Model |
|------|-------|--------------|
| 1 | Prompt Generation | Python + LLM-style templates |
| 2 | Synthetic Image Generation | Stable Diffusion SDXL Turbo |
| 3 | Automated Annotation | Segment Anything Model (SAM ViT-H) |
| 4 | Dataset Preparation | YOLO format, 80/20 train/val split |
| 5 | Object Detection Training | YOLOv8n (`yolov8n.pt`) |
| 6 | Segmentation Training | YOLOv8n-seg (`yolov8n-seg.pt`) |
| 7 | Evaluation | F1, Precision, Recall, mAP curves |
| 8 | Image Captioning | Salesforce/blip-image-captioning-large |

---

## Models Used

| Model | Purpose |
|-------|---------|
| `stabilityai/sdxl-turbo` | Text-to-image synthetic data generation |
| `facebook/sam-vit-h` (SAM ViT-H) | Automated segmentation mask generation |
| `ultralytics/yolov8n` | Object detection (CSPDarknet53 + PANet) |
| `ultralytics/yolov8n-seg` | Instance segmentation |
| `Salesforce/blip-image-captioning-large` | Scene captioning (Encoder-Decoder + LSTM attention) |

---

## Dataset

- **Classes:** vehicles, pedestrians, traffic_signs, road_conditions, animals
- **Images generated:** 20 (configurable via `N_IMAGES_TO_GENERATE`)
- **Train / Val split:** 16 train | 4 val (80/20)
- **Annotation format:** YOLO `.txt` (bounding box + segmentation polygon)
- **Scenarios:** Day/night, clear/rain/fog/snow, urban India / Europe / US suburb / highway

---

## Results

### Detection (Bounding Box)
| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| all | 0.049 | 0.800 | 0.383 | 0.298 |
| vehicles | 0.058 | 0.900 | 0.443 | 0.335 |
| pedestrians | 0.040 | 0.700 | 0.324 | 0.261 |

> Note: Low precision is expected at this scale (20 images). Increase `N_IMAGES_TO_GENERATE` to 500+ for production-level metrics matching the paper.

### Image Generation Quality (FID Score — from paper)
| Model | FID Score ↓ |
|-------|-------------|
| **Diffusion Models (Ours)** | **2.27** |
| GAN | 2.84 |
| VAE | 31.11 |

---

## Requirements

```
Python 3.12
torch >= 2.0
diffusers
transformers
accelerate
ultralytics
segment-anything
opencv-python-headless
matplotlib
Pillow
huggingface_hub
```

Install all:
```bash
pip install diffusers transformers accelerate ultralytics segment-anything \
            opencv-python-headless matplotlib Pillow huggingface_hub
```

---

## Setup & Usage

### 1. Open in Google Colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

> **Runtime:** T4 GPU (free) for steps 1–3 and 8. A100 (Colab Pro) recommended for step 5–6 (YOLOv8 training).

### 2. Download SAM Checkpoint
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
The notebook handles this automatically in Cell 1.

### 3. Run All Cells in Order
Each cell is self-contained and labelled by phase. Run top to bottom.

### 4. Scale Up (Optional)
In Cell 5, increase the image count:
```python
N_IMAGES_TO_GENERATE = 500   # default is 20
```
In Cell 13, increase training epochs:
```python
EPOCHS = 100   # default is 50
```

---

## Output Files

After running all cells, `synthetic_detection_outputs.zip` is auto-downloaded containing:

| File | Description |
|------|-------------|
| `runs/detect/.../best.pt` | Best detection model weights |
| `runs/segment/.../best.pt` | Best segmentation model weights |
| `generated_images_grid.png` | Grid of SDXL Turbo generated images |
| `annotated_images.png` | SAM annotation visualization |
| `class_distribution.png` | Training set class distribution chart |
| `evaluation_curves.png` | F1 + Precision-Confidence curves (Figures 7–10) |
| `fid_comparison.png` | FID score comparison bar chart (Table 2) |
| `detection_results.png` | YOLOv8 inference on val images |
| `pipeline_output.png` | Side-by-side original + segmented + caption |
| `prompts/all_prompts.json` | All generated prompts |
| `synthetic_images/metadata.json` | Generated image metadata |

---

## Project Structure

```
.
├── Object_Detection_Synthetic_Data.ipynb   # Main notebook
├── prompts/
│   └── all_prompts.json
├── synthetic_images/
│   ├── metadata.json
│   └── *.jpg
├── annotations/
│   ├── images/
│   ├── labels/          # YOLO bbox format
│   └── seg_labels/      # YOLO segmentation format
├── dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   ├── dataset_detect.yaml
│   └── dataset_segment.yaml
├── runs/
│   ├── detect/
│   └── segment/
└── sam_vit_h_4b8939.pth
```

---

## Reference

```
O. Pavan Kumar, K. Aditya Vardhan, Bhukya Chinni Krishna, Palakonda Jesse Angelina,
Bollapalli Althaph, Beebi Naseeba.
"Object Detection using Synthetic Data and Generative AI"
Proceedings of ICSCDS-2025, IEEE.
DOI: 10.1109/ICSCDS65426.2025.11167528
```

---

## GPU Requirements

| Task | Minimum GPU | Recommended |
|------|-------------|-------------|
| Image Generation (SDXL Turbo) | T4 (16GB) | A100 |
| SAM Annotation | T4 (16GB) | T4 |
| YOLOv8 Training | A100 (Colab Pro) | A100 |
| BLIP Captioning | T4 (16GB) | T4 |
