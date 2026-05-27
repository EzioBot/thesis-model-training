# Bangla License Plate Detection with YOLOv8

This project trains a YOLOv8 object detector to locate full Bangla license
plates. The notebook creates a normalized one-box-per-plate dataset before
creating augmented training data and training the detector. It also supports
source annotations that contain multiple character boxes by merging them into
one plate box.

## Workflow

The notebook performs these steps:

1. Checks the available CPU or NVIDIA GPU device.
2. Verifies the downloaded dataset structure.
3. Converts or normalizes annotations into one plate-level box per image.
4. Creates clean and degraded images using blur, distortion, and lighting
   augmentation.
5. Trains a YOLOv8 model to detect one class: `license_plate`.
6. Validates and tests the best trained model.
7. Displays metrics, plots, and final prediction examples.

## Project Files

| File | Description |
| --- | --- |
| `yolov8_workflow.ipynb` | Main preprocessing, training, and evaluation notebook. |
| `yolov8s.pt` | Pretrained YOLOv8 weights used to begin training. |
| `commands .docx` | Original Windows environment setup command reference. |

## Requirements

- Windows PowerShell or the VS Code PowerShell terminal
- Python available from the `python` command
- An NVIDIA GPU is recommended for training, but the notebook can fall back
  to CPU

## Step 1: Open the Project Folder

Open this folder in VS Code, and then open **Terminal > New Terminal**. For
the current local folder, PowerShell can also be opened directly with:

```powershell
cd "D:\thesis model training"
```

## Step 2: Create a Python Virtual Environment



```powershell
python -m venv --clear .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

After activation, the terminal prompt should begin with `(.venv)`.

If PowerShell prevents virtual-environment activation, run the following
command once and activate the environment again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

## Step 3: Install Required Packages

For an NVIDIA GPU environment using the CUDA 12.8 PyTorch build specified in
the setup document, run:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install jupyterlab numpy pandas matplotlib scikit-learn
python -m pip install ultralytics albumentations opencv-python
```

For CPU-only training, use the CPU PyTorch packages instead of the CUDA line:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install jupyterlab numpy pandas matplotlib scikit-learn
python -m pip install ultralytics albumentations opencv-python
```

## Step 4: Download and Arrange the Dataset

The primary source dataset is hosted on Roboflow Universe under the CC BY 4.0
license and is listed there as object detection data with the
`license-plate-number` class:

- [Bangla License Plate Detection - Roboflow Universe](https://universe.roboflow.com/bangla-car-license-plate-detection/bangla-license-plate-detection-p3xyb)
- [Bangla License Plate Detection v1 - versioned dataset export](https://universe.roboflow.com/sagar-sarker-4fqam/bangla-license-plate-detection-p3xyb-unrnf/dataset/1)

Download the dataset in **YOLOv8** format. Extract its contents into a
`dataset/` folder in the project root. Before running the notebook, the
required structure is:

```text
dataset/
|-- train/
|   |-- images/
|   `-- labels/
|-- valid/
|   |-- images/
|   `-- labels/
|-- test/
|   |-- images/
|   `-- labels/
`-- data.yaml
```

The notebook writes a plate-level dataset using the class name
`license_plate`. If source annotations contain multiple character boxes, it
merges them into one full-plate bounding box automatically. If an annotation
already contains one plate box, that plate-level structure is preserved.

## Step 5: Run the Notebook

With the virtual environment active, start JupyterLab:

```powershell
jupyter lab
```

Open `yolov8_workflow.ipynb` and run the cells in order. The notebook uses
these default training settings:

| Setting | Value |
| --- | --- |
| Model weights | `yolov8s.pt` |
| Epochs | `30` |
| Image size | `640` |
| Batch size | `8` |

If GPU memory is limited, reduce `BATCH_SIZE` to `4` in the training cell.

## Step 6: Review Generated Outputs

Running the notebook creates these local folders:

| Folder | Contents |
| --- | --- |
| `dataset_plate/` | Images and converted one-box license-plate labels. |
| `dataset_processed/` | Clean and augmented plate-detection data used for training. |
| `runs/bangla_plate_yolov8/` | Trained weights, validation plots, metrics, and predictions. |

The best trained model is saved under:

```text
runs/bangla_plate_yolov8/plate_motion_blur_distortion_exp/weights/best.pt
```

## Version Control

Downloaded datasets, dataset archives, converted/generated dataset folders,
and training run outputs are excluded from Git. The pretrained weight files
used by this repository are included so the notebook can start from the
documented model weights.

## License

The project source is provided under the [MIT License](LICENSE). Refer to the
Roboflow dataset page for the dataset license and attribution requirements.
