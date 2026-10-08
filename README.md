# Placenta MRI Segmentation (U-Net)

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C.svg)

## Project Overview
Placenta Accreta Spectrum (PAS) is a severe pregnancy complication where the placenta attaches too deeply into the uterine wall. Early and accurate detection via MRI is critical for maternal safety. 

This project implements a **High-Capacity Convolutional Neural Network (U-Net)** to automatically segment and identify the placenta in medical MRI scans. By automating this radiological process, the model assists in identifying the spatial boundaries of the placenta, demonstrating the application of Computer Vision in healthcare.

## Dataset
The model was trained on the publicly available **Placenta Accreta Spectrum Disorders (PASDs) MRI Dataset**.
* **Source:** [Mendeley Data - PASDs](https://data.mendeley.com/datasets/284gwmf5bh/1)
* **Format:** Ground truth radiological masks paired with grayscale MRI slices.

## Architecture
The core architecture is a custom, widened **U-Net** built using **PyTorch** and **MONAI**.
* **Frameworks:** PyTorch, MONAI, Matplotlib, PIL
* **Model Depth:** 5 layer deep Encoder/Decoder `(32, 64, 128, 256, 512 channels)`
* **Processing Blocks:** 4 Residual Units per layer for complex tissue texture extraction.
* **Loss Function:** Dice Loss (Optimized for spatial overlap over pixel wise accuracy)
* **Optimizer:** Adam (`lr=1e-5`)

## Methodology
To ensure the model learns generalized anatomical features rather than memorizing the training data, several enterprise-grade techniques were implemented:
1. **Dynamic Data Augmentation:** Applied random horizontal/vertical flips and ±15° rotations on the fly during training to create an infinitely variable dataset.
2. **Train/Validation Split:** The dataset was strictly split into an 80% Training set and a 20% unseen Validation Vault to actively monitor for and prevent overfitting.
3. **Regularization:** A 20% Dropout rate was applied across the network to force redundant feature learning.

## Overlay Images

<img width="512" height="256" alt="best_sub082_31" src="https://github.com/user-attachments/assets/60c1ae36-11e2-4ea6-94f6-9bc95d81ad7e" />
<img width="512" height="256" alt="best_sub082_35" src="https://github.com/user-attachments/assets/cbea8671-c185-4ded-ab94-f6bdfd2cc150" />
<img width="512" height="256" alt="best_sub089_24" src="https://github.com/user-attachments/assets/44093c2a-3f49-49f8-8f6c-dcea42dd0090" />


## Results
| Method | Dice Similarity Coefficient (DSC) | Intersection over Union (IoU) |
| :--- | :---: | :---: |
| **Otsu Baseline (Per-Sequence)** | 0.329 | --- |
| **Proposed Method (Per-Sequence)** | **0.860** | **0.770** |

## Limitations
Single run and seed, small model (482k parameters), 52.3% connected-component agreement, no external validation.



*Developed by Lakshya Arora for Deep Learning & Medical Imaging research.*
