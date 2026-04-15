# Comparative Analysis of CNN Architectures for Brain MRI Classification using Two-Stage Transfer Learning with Grad-CAM Interpretation

A deep learning research project performing a systematic comparison of three CNN architectures — VGG16, ResNet50, and EfficientNetB0 — for multiclass brain tumor classification from MRI scans, using a two-stage transfer learning strategy and Grad-CAM visualization for clinical interpretability.

---

## Motivation

Brain tumor diagnosis from MRI scans is a critical and time-sensitive clinical task. Deep learning models have shown strong potential in medical image analysis, but selecting the right architecture for deployment requires rigorous benchmarking under identical experimental conditions. This project evaluates three widely-used CNN architectures to identify the most effective model for brain tumor classification while maintaining clinical explainability through Grad-CAM.

---

## Project Highlights

- **3 CNN architectures** benchmarked under identical training conditions
- **Two-stage transfer learning** — frozen backbone training followed by selective last-block fine-tuning
- **Functional block equivalence** — Block5 (VGG16), Conv5 (ResNet50), Block7 (EfficientNetB0) unfrozen for fair comparison
- **Per-class confusion matrices** across all three models
- Discovered that **EfficientNetB0 resists fine-tuning** due to compound-scaled SE block sensitivity — a research-worthy finding

---

## Dataset

[Brain Tumor MRI Dataset — Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

- 4 classes: **Glioma, Meningioma, Pituitary Tumor, No Tumor**
- MRI scan images preprocessed and augmented for training
- Train/Validation split with data augmentation

---

## Two-Stage Transfer Learning Strategy

```
Stage 1 — Feature Extraction
   Backbone frozen (ImageNet weights)
   Only classification head trained
   ↓
Stage 2 — Fine-Tuning
   Final convolutional block unfrozen
   Entire network trained at low LR (1e-5)
```

**Block unfrozen per model:**

| Model | Unfrozen Block | Equivalent |
|---|---|---|
| VGG16 | block5 | Last conv block |
| ResNet50 | conv5 | Last residual stage |
| EfficientNetB0 | block7a | Last MBConv block |

---

## Results

### Stage-wise Performance

| Model | Stage 1 Val Acc | Stage 1 Val Loss | Stage 2 Val Acc | Stage 2 Val Loss |
|---|---|---|---|---|
| VGG16 | 91.61% | 0.2559 | **96.07%** | **0.1289** |
| ResNet50 | 93.30% | 0.1806 | **96.34%** | **0.0984** |
| EfficientNetB0 | **94.46%** | **0.1504** | 93.66% | 0.1862 |

---

### 🔹 Final Model Comparison

| Model | Test Accuracy | Macro F1 | Selected Stage |
|------|-------------|----------|----------------|
| Baseline CNN | 86% | 0.86 | — |
| VGG16 | 92% | 0.92 | Stage 2 |
| ResNet50 | **92%** | **0.92** | Stage 2 |
| EfficientNetB0 | 91% | 0.90 | Stage 1 |


---

## Key Observations

1. **VGG16 and ResNet50** showed clear improvement in Stage 2, confirming that selective last-block unfreezing enables effective task-specific feature adaptation for medical imaging.

2. **EfficientNetB0** exhibited a unique behavior — Stage 2 fine-tuning degraded performance compared to Stage 1. This is attributed to the Squeeze-and-Excitation (SE) attention mechanism in block7, which has a global receptive field and is more sensitive to fine-tuning than localized conv filters. Stage 1 EfficientNetB0 (94.46%) was selected as its representative result.

3. **EfficientNetB0 Stage 1 outperformed both VGG16 and ResNet50 Stage 1**, demonstrating that compound-scaled pretrained features transfer more effectively to medical imaging with a frozen backbone.


---
### 🔍 Hardest Class: Glioma
- Lowest recall across models (~0.70–0.78)
- High similarity with other tumor classes


###  Best Model

**ResNet50 (Stage 2)**
- Best overall performance
- Achieved the highest validation accuracy (96.34%)
- Demonstrated stable convergence during fine-tuning
- Balanced accuracy and efficiency

##  Evaluation Metrics

- Confusion Matrix
- Precision / Recall
- Macro F1-Score
- Classification Report


## 🧪 Explainability (Grad-CAM)

- Generated class activation maps
- Visualized tumor regions
- Improved interpretability for medical use


## Technologies

`Python` `TensorFlow` `Keras` `OpenCV` `NumPy` `Matplotlib` `Seaborn` `Jupyter Notebook`



## ⭐ Star this repo if you found it useful!
---


---
