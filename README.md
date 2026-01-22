# OCR System Using YOLO and OCR Models

## 1. Introduction

This project implements an end-to-end Optical Character Recognition (OCR) system that combines:

- **YOLO** for text line detection in images and videos
- **OCR model** for text recognition from the detected text regions

The system follows a two-stage pipeline:

1. Detect text regions using YOLO
2. Crop detected regions and recognize text using an OCR model

This design allows the detection and recognition components to be trained, optimized, and replaced independently, making the system flexible and scalable for real-world OCR applications.

## 2. System Pipeline

```
Input Image / Video
        │
        ▼
YOLO (Text Line Detection)
        │
        ▼
Crop Detected Text Regions
        │
        ▼
OCR Model (Text Recognition)
        │
        ▼
Recognized Text Output
```

## 3. Project Structure

```
.
├── Create_OCR_data.py              # Generate OCR training dataset
├── Create_Yolo_data.py             # Generate YOLO training dataset
├── create_ocr_data.yaml            # OCR data generation config
├── create_yolo_data_config.yaml    # YOLO data generation config
├── Train_ocr_from_scratch.py       # Train OCR model from scratch
├── Train_yolo.py                   # Train YOLO model
├── train_ocr_config.yaml           # OCR training config
├── train_yolo_data.yaml            # YOLO training config
├── System_Inference.py             # End-to-end inference pipeline
│
├── model/                          # Trained models
├── ocr_dataset/                    # OCR dataset
├── runs/                           # Training logs and outputs
├── SceneTrialTrain/                # YOLO dataset
├── video/                          # Demo videos
└── README.md
```

## 4. Installation

### 4.1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/VyDat-1702/OCR-YOLOv11.git
cd  OCR-YOLOv11/
```

### 4.2. Requirements

- Python ≥ 3.10
- Conda (recommended)
- NVIDIA GPU with CUDA support (optional but recommended)

### 4.3. Environment Setup

Create and activate a conda environment:

```bash
conda create -n pytorch_env python=3.12
conda activate pytorch_env
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 5. Dataset Preparation

### 5.1. Create YOLO Dataset (Text Detection)

```bash
python Create_Yolo_data.py --config create_yolo_data_config.yaml
```

This script prepares the dataset for training the YOLO text detection model.

### 5.2. Create OCR Dataset (Text Recognition)

```bash
python Create_OCR_data.py --config create_ocr_data.yaml
```

This script generates cropped text images and corresponding labels for OCR training.

## 6. Model Training

### 6.1. Train YOLO (Text Detection)

```bash
python Train_yolo.py --config train_yolo_data.yaml
```

Training results (weights, logs) are saved in: `runs/`

### 6.2. Train OCR Model

```bash
python Train_ocr_from_scratch.py --config train_ocr_config.yaml
```

The trained OCR model is saved in the `model/` directory.

## 7. Inference (Run the Full System)

Run the complete OCR pipeline on images or videos:

```bash
python System_Inference.py
```

**Output:**

- Images or videos with detected text regions
- Recognized text results for each detected region

## 8. Notes

- It is not recommended to commit large trained model files (`.pt`) directly to the repository.
- Use Git LFS or download trained models from releases or external storage if needed.
- Ensure you have sufficient disk space for datasets and training outputs.
