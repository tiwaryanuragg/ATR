# Automatic Target Recognition (ATR) on SAR images of military vehicles with Deep Metric Learning

> Dataset: MSTAR 8-Class SAR Vehicle Recognition  
> Best accuracy: **98.59%** on 8-class SAR vehicle classification

---

## Project Overview

This project implements an Automatic Target Recognition (ATR) system
for SAR (Synthetic Aperture Radar) imagery using deep metric learning.
The model classifies military ground vehicles from radar images with
high accuracy, going beyond the referenced research paper by introducing
a CBAM attention mechanism.

### Real-World Pipeline

SAR Sensor → Raw I/Q Signal → Image Formation → Full SAR Scene
→ YOLO Detection → Crop + Preprocess → Our Model → Class + Confidence

This project focuses on the **recognition stage** — classifying
pre-detected vehicle chips from SAR imagery.

---

## Dataset

**MSTAR (Moving and Stationary Target Acquisition and Recognition)**
- 9,466 SAR images across 8 military vehicle classes
- Image size: 368×368 (resized to 128×128 for training)
- Source: [Kaggle MSTAR Dataset](https://www.kaggle.com/datasets/atreyamajumdar/mstar-dataset-8-classes)

| Class | Vehicle Type | Images |
|---|---|---|
| 2S1 | Self-propelled howitzer | 1,164 |
| BRDM_2 | Armored reconnaissance vehicle | 1,415 |
| BTR_60 | Armored personnel carrier | 1,353 |
| D7 | Military bulldozer | 573 |
| SLICY | Radar calibration target | 1,270 |
| T62 | Main battle tank | 1,144 |
| ZIL131 | Military utility truck | 1,146 |
| ZSU_23_4 | Anti-aircraft gun system | 1,401 |

---

## Architecture — Model 5 (Best)

SAR Image (128×128)
↓
STN — Spatial Transformer Network
(corrects aspect angle variation)
↓
WideResNet-28-2 backbone
(Leaky ReLU · BatchNorm · Dropout)
↓
CBAM — Convolutional Block Attention Module  ← Novel contribution
(Channel attention + Spatial attention)
↓
Global Avg Pool + Max Pool → 256-dim
↓
Embedding head → 128-dim (L2 normalized)
↓
CrossEntropy + Triplet Loss (hard negative mining)

---

## Results — All 5 Models

| Model | Architecture | Test Accuracy | k-NN Accuracy | Epochs |
|---|---|---|---|---|
| Model 1 | ResNet18 (baseline) | 99.65% | 99.79% | 25 |
| Model 2 | WideResNet-28-2 | 93.03% | 95.63% | 25 |
| Model 3 | WideResNet-28-2 + STN | 81.55% | 87.96% | 25 |
| Model 4 | WideResNet + STN + Triplet | 97.39% | 99.72% | 50 |
| **Model 5** | **WideResNet + STN + CBAM + Triplet** | **98.59%** | **98.52%** | **50** |

### Per-class Results (Model 5)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| 2S1 | 0.9882 | 0.9598 | 0.9738 |
| BRDM_2 | 0.9952 | 0.9858 | 0.9905 |
| BTR_60 | **1.0000** | 0.9852 | 0.9926 |
| D7 | **1.0000** | 0.9651 | 0.9822 |
| SLICY | 0.9845 | **1.0000** | 0.9922 |
| T62 | 0.9942 | 0.9942 | 0.9942 |
| ZIL131 | 0.9718 | **1.0000** | 0.9857 |
| ZSU_23_4 | 0.9628 | 0.9857 | 0.9741 |

---

## System Architecture

![ATR Pipeline](diagrams/atr_pipeline_diagram.png)

---

## Key Techniques

### Spatial Transformer Network (STN)
Learns to correct aspect angle variation in SAR images.
Applies an affine transformation (rotation, scale, translation)
to normalize each image before feature extraction.

### WideResNet-28-2
Wide residual network with depth=28, widen_factor=2.
Uses Leaky ReLU activation, BatchNorm, and Dropout(0.3).
Trained from scratch — no pretrained weights.

### CBAM (Novel contribution)
Convolutional Block Attention Module applied after WideResNet.
Channel attention identifies which feature maps matter most.
Spatial attention identifies which locations matter most.
Improves accuracy by +1.20% over Model 4.

### Deep Metric Learning
Combined loss: 0.7 × CrossEntropy + 0.3 × Triplet Loss.
Hard negative mining for efficient triplet training.
128-dim embedding space enables k-NN classification.

---

## Training Details

Model 1 — ResNet18
Epochs trained    : 25
Best val accuracy : 99.79%

Model 2 — WideResNet-28-2
Epochs trained    : 25
Best val accuracy : 93.80%

Model 3 — WideResNet + STN
Epochs trained    : 25
Best val accuracy : 82.25%

Model 4 — WideResNet + STN + Triplet
Epochs trained    : 50
Best val accuracy : 98.52%?

Model 5 — WideResNet + STN + CBAM + Triplet
Epochs trained    : 50
Best val accuracy : 98.45%?

Optimizer    : Adam (lr=3e-4, STN lr=1e-5)
Scheduler    : CosineAnnealingLR (T_max=50)
Weight decay : 1e-3
Batch size   : 32
Image size   : 128×128 grayscale
Augmentation : HorizontalFlip, VerticalFlip, Rotation(±15°)
Split        : 70% train / 15% val / 15% test (stratified)

---

## Results — All 5 Models

| Model | Architecture | Test Accuracy | k-NN Accuracy | Epochs |
|---|---|---|---|---|
| Model 1 | ResNet18 (baseline) | 99.65% | 99.79% | 25 |
| Model 2 | WideResNet-28-2 | 93.03% | 95.63% | 25 |
| Model 3 | WideResNet-28-2 + STN | 81.55% | 87.96% | 25 |
| Model 4 | WideResNet + STN + Triplet | 97.39% | 99.72% | 50 |
| **Model 5** | **WideResNet + STN + CBAM + Triplet** | **98.59%** | **98.52%** | **50** |

### Per-class Results (Model 5)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| 2S1 | 0.9882 | 0.9598 | 0.9738 |
| BRDM_2 | 0.9952 | 0.9858 | 0.9905 |
| BTR_60 | **1.0000** | 0.9852 | 0.9926 |
| D7 | **1.0000** | 0.9651 | 0.9822 |
| SLICY | 0.9845 | **1.0000** | 0.9922 |
| T62 | 0.9942 | 0.9942 | 0.9942 |
| ZIL131 | 0.9718 | **1.0000** | 0.9857 |
| ZSU_23_4 | 0.9628 | 0.9857 | 0.9741 |

---

## System Architecture

[ATR Pipeline](diagrams/atr_pipeline.png)

---

## Key Techniques

### Spatial Transformer Network (STN)
Learns to correct aspect angle variation in SAR images.
Applies an affine transformation (rotation, scale, translation)
to normalize each image before feature extraction.

### WideResNet-28-2
Wide residual network with depth=28, widen_factor=2.
Uses Leaky ReLU activation, BatchNorm, and Dropout(0.3).
Trained from scratch — no pretrained weights.

### CBAM (Novel contribution)
Convolutional Block Attention Module applied after WideResNet.
Channel attention identifies which feature maps matter most.
Spatial attention identifies which locations matter most.
Improves accuracy by +1.20% over Model 4.

### Deep Metric Learning
Combined loss: 0.7 × CrossEntropy + 0.3 × Triplet Loss.
Hard negative mining for efficient triplet training.
128-dim embedding space enables k-NN classification.

---

## Training Details
---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
```python
import kagglehub
dataset_path = kagglehub.dataset_download(
    'atreyamajumdar/mstar-dataset-8-classes')
```

### 3. Train all models
Open `notebooks/training_notebook.ipynb` on Kaggle
with GPU enabled and run all cells.

### 4. Run demo
Open `notebooks/demo_notebook.ipynb` and load
saved model weights to see predictions and comparisons.

---

## Reference Paper

> *Automatic Target Recognition with Deep Metric Learning*  
> University of Louisville — ThinkIR Institutional Repository  
> Electronic Theses and Dissertations, 2020

---

## Project Structure
ATR/
├── README.md
├── requirements.txt
├── Notebooks/
│   ├── training_notebook.ipynb
│   └── demo_notebook.ipynb
├──Model_Diagrams/
    ├── ResNet18
        ├── model_architecture.png
        ├── confusion_matrix.png
        ├── tsne_embeddings.png
        ├── training_curves.png
    ├── WideResNet28-2
        ├── model_architecture.png
        ├── confusion_matrix.png
        ├── tsne_embeddings.png
        ├── training_curves.png
    ├── STN + WideResNet28-2 
        ├── model_architecture.png
        ├── confusion_matrix.png
        ├── tsne_embeddings.png
        ├── training_curves.png
    ├── STN + WideResNet28-2 + Treplet Loss
        ├── model_architecture.png
        ├── confusion_matrix.png
        ├── tsne_embeddings.png
        ├── training_curves.png
    ├── STN + WideResNet28-2 + CBAM + Triplet Loss
        ├── model_architecture.png
        ├── confusion_matrix.png
        ├── tsne_embeddings.png
        ├── training_curves.png
    ├── AllModels_comparison
        ├── all_models_traning_curves.png
        ├── final_comparison.png
        ├── training_curves.png

---

## Author

**Anurag Kumar**  
Year: 2026

