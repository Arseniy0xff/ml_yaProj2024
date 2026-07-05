# ml_yaProj2024 (Human Pose Classification)

PyTorch solution for classifying human activity/pose images into 20 categories for the ML Intensive Yandex Academy Autumn 2024 competition.

## Overview

This project trains a custom CNN with:
- residual/skip connections
- Squeeze-and-Excitation blocks
- dropout and batch normalization
- data augmentation
- macro F1 as the main validation metric

It also generates a `submission.csv` file for competition upload.

## Dataset

Expected dataset layout after extraction:

```text
/content/Dataset/human_poses_data/
├── img_train/
├── img_test/
├── train_answers.csv
└── activity_categories.csv
```

### CSV files
- `train_answers.csv` — training image IDs and labels
- `activity_categories.csv` — class ID to category mapping

## Model

The model is a custom CNN:
- several `SkipConnectionBlock` stages
- `SEBlock` attention modules
- max pooling between stages
- adaptive average pooling at the end
- MLP classifier head

### Input
- RGB images
- resized to `256 x 256`

### Output
- 20-class classification logits

## Training

### Loss
- `CrossEntropyLoss`

### Optimizer
- `Adam`
- learning rate: `0.001`

### Metric
- `macro F1 score`

### Augmentation
Used on one copy of the training set:
- horizontal flip
- color jitter
- small rotation

The original and augmented datasets are concatenated to increase training diversity.

## Inference

During inference:
- the best checkpoint is loaded
- predictions are made on test images
- results are saved in `submission.csv`

## Output Format

`submission.csv` contains:

```text
id,target_feature
```

## Requirements

```bash
pip install scikit-learn Pillow torchvision matplotlib pandas tqdm torch
```

## How to Run

1. Open the notebook in Google Colab.
2. Upload `kaggle.json`.
3. Download the competition dataset.
4. Train the model.
5. Save the best checkpoint.
6. Run inference and generate `submission.csv`.

## Notes

- The notebook is designed for Google Colab.
- GPU is recommended.
- The model checkpoint is saved when validation macro F1 improves.
