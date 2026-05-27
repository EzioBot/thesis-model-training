# Bangla License Plate Detection Model Training

This repository contains the Jupyter notebook for training a YOLOv8 model to
detect Bangla license plates. Dataset files, model weights, and generated
training results are intentionally excluded from GitHub so that the repository
contains the reproducible training workflow rather than large data artifacts.

## Code

- `yolov8_workflow.ipynb` - model training and evaluation workflow.

## Dataset

The dataset is hosted on Roboflow Universe under the CC BY 4.0 license:

- [Bangla License Plate Detection v1](https://universe.roboflow.com/sagar-sarker-4fqam/bangla-license-plate-detection-p3xyb-unrnf/dataset/1)

Download the dataset in YOLOv8 format before running the notebook. Local
dataset folders such as `dataset/`, `dataset_plate/`, and
`dataset_processed/` are ignored by Git.

## Excluded Artifacts

Generated training outputs in `runs/`, downloaded or trained model weights,
and local dataset archives are excluded from version control.
