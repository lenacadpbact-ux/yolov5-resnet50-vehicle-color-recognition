# Confidence-Gated Vehicle Colour Recognition

**YOLOv5 Detection with ResNet-50 Fallback Classification**

A two-stage deep learning pipeline for vehicle colour recognition that combines a YOLOv5 detector with a ResNet-50 classifier through a confidence-gated fusion policy. When YOLOv5's detection confidence falls below τ = 0.60, the cropped vehicle region is escalated to ResNet-50 for a fine-grained colour decision — reducing misclassification in difficult lighting and environmental conditions.

## Overview

- **Stage 1 — Detection (YOLOv5):** Localises vehicles and produces an initial colour prediction with a confidence score.
- **Stage 2 — Classification (ResNet-50):** Invoked only when YOLOv5's confidence is below the fusion threshold (τ = 0.60); classifies a 128×128 crop of the detected region.
- **Fusion policy:** Each final prediction is tagged with its source model (`YOLO` or `ResNet`), so every decision is traceable.

## Dataset

[Cars Colour Recognition Dataset (v2)](https://universe.roboflow.com/final-project-jwpes/cars-color-recognition/dataset/2) — 5,077 images across 6 colour classes: `black`, `blue`, `green`, `red`, `white`, `yellow`.

| Split | Images |
|-------|--------|
| Train | 4,461 |
| Val   | 406 |
| Test  | 210 |

## Results

Peak validation accuracy: **98.28%** (Epoch 44, ResNet-50)

| Class | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|-----------|--------|---------|--------------|
| All   | 406 | 0.980 | 0.971 | 0.988 | 0.985 |
| Black | 83  | 1.000 | 0.952 | 0.995 | 0.991 |
| Red   | 85  | 1.000 | 0.991 | 0.995 | 0.995 |
| Blue  | 60  | 0.967 | 0.964 | 0.992 | 0.992 |
| White | 62  | 0.975 | 0.968 | 0.991 | 0.983 |
| Green | 64  | 0.983 | 0.969 | 0.981 | 0.979 |
| Yellow| 52  | 0.955 | 0.981 | 0.971 | 0.971 |

## Pipeline Steps

The full pipeline is implemented in [`vehicle_colour_pipeline.py`](./vehicle_colour_pipeline.py), organised as:

| Step | Description |
|------|-------------|
| 1 | Environment & reproducibility check |
| 2 | Install YOLOv5 + dependencies |
| 3 / 3.1 / 3.2 | Load dataset YAML, normalize paths, sanity-check labels |
| 4 | Train YOLOv5 (50 epochs) |
| 5 | Validate YOLOv5 (mAP/PR) and extract anchors |
| 6 | Build CNN dataset (folder-per-colour crops from YOLO labels) |
| 7 | Train ResNet-50 (50 epochs) |
| 8 | Unified fusion inference (single-image & batch modes) |
| 9 | MATLAB convergence plot |

## Requirements

- Python 3.10+
- PyTorch, torchvision
- [YOLOv5](https://github.com/ultralytics/yolov5) (cloned automatically by the script)
- `torchmetrics==1.3.0`, `seaborn==0.13.2`
- OpenCV (`cv2`), pandas, matplotlib, PyYAML
- MATLAB R2025b (for the convergence plot in Step 9, optional)

Designed to run on Google Colab with a Google Drive–mounted dataset, but adaptable to any environment with the paths adjusted.

## Usage

1. Mount your dataset (`data.yaml`) at the path referenced in Step 3, or update the path to match your own location.
2. Run Steps 1–2 to set up the environment and clone YOLOv5.
3. Run Step 3 to normalize the dataset YAML.
4. Run Step 4 to train YOLOv5, then Step 5 to validate.
5. Run Step 6 to build the cropped CNN dataset from YOLO labels.
6. Run Step 7 to train ResNet-50 on the cropped dataset.
7. Run Step 8 to perform fused inference (single image or batch) using both trained models.

## Limitations

Sensitive to extreme lighting, particularly for `white` and `yellow` vehicles, where background interference reduces classification confidence. The current pipeline handles one primary detection per image; multi-object detection/classification in a single frame is planned future work.

## References

- Çaldıran, O., & Acarman, T. (2024). Vehicle attribute and licence plate recognition with deep learning based hierarchical pipeline. *IEEE Intelligent Vehicles Symposium*, pp. 1–8.
- Final Project JWPES. (2023). Cars colour recognition dataset (v2). Roboflow Universe.
- Hassan, A., Ali, M., Durrani, M. N., & Tahir, M. A. (2022). Vehicle recognition using multi-level deep learning models. *International Conference on Engineering Software for Modern Challenges*, pp. 101–113.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *IEEE Conference on Computer Vision and Pattern Recognition*, pp. 770–778.
- Jocher, G. (2020). Ultralytics YOLOv5 (Version 7.0) [Software]. Zenodo.
- Khanam, R., & Hussain, M. (2024). What is YOLOv5: A deep look into the internal features of the popular object detector. *arXiv*.
- Kim, J. (2024). Deep learning-based vehicle type and color classification to support safe autonomous driving. *Applied Sciences*, 14(4), pp. 1–17.
- Lima, G. E., Laroca, R., Santos, E., Nascimento Jr., E., & Menotti, D. (2024). Toward enhancing vehicle colour recognition in adverse conditions: A dataset and benchmark. *Conference on Graphics, Patterns and Images (SIBGRAPI)*, pp. 1–6.

## Author

Arlen Balunan
