# Multi-QR Code Recognition Hackathon Submission

This repository contains the source code for the multi-QR code recognition hackathon. The solution uses a YOLOv8 model for QR code detection and `pyzbar` for decoding.

## Repository Structure

```
multiqr-hackathon/
│
├── README.md
├── requirements.txt
├── train.py
├── infer.py
│
├── data/
│   ├── QR_DS/            # (Not committed) Your dataset should be placed here
│   │   ├── train_images/
│   │   ├── valid_images/
│   │   └── test_images/
│   └── data.yaml         # Dataset configuration file
│
└── outputs/
    ├── submission_detection_1.json
    └── submission_decoding_2.json
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd multiqr-hackathon
```

### 2. Environment Setup

It is recommended to use a Python virtual environment.

```bash
# Create a virtual environment (e.g., using venv)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install system dependencies (for pyzbar)
sudo apt-get update
sudo apt-get install -y libzbar0

# Install Python packages
pip install -r requirements.txt
```

### 3. Data Preparation

1.  Place your dataset folder (e.g., `QR_DS`) inside the `data/` directory.
2.  Create a `data.yaml` file inside the `data/` directory with the correct paths to your training and validation sets. **You must update the paths below to match your local setup.**

    ```yaml
    # data/data.yaml
    train: /path/to/your/project/data/QR_DS/train_images/images
    val: /path/to/your/project/data/QR_DS/valid_images/images

    # Number of classes
    nc: 1

    # Class names
    names: ['qr_codes']
    ```

## How to Run

### Training

To train the model from scratch, run the `train.py` script. The trained model weights (`best.pt`) will be saved in the `runs/detect/qr_yolo_model/` directory.

```bash
python train.py --data_yaml data/data.yaml --epochs 100

```

### Training

To train the model from scratch, run the `train.py` script. The trained model weights (`best.pt`) will be saved in the `runs/detect/qr_yolo_model/` directory.

You can customize the training parameters, such as epochs and patience.

**Example:**
```bash
python train.py \
    --data_yaml data/data.yaml \
    --epochs 100 \
    --patience 20
###

### Inference

The `infer.py` script runs the detection and decoding process. It requires the path to the trained model, the input directory of images, the desired output file, and the submission stage.

**Stage 1: Detection Only**

This will generate the `submission_detection_1.json` file required for the mandatory task.

```bash
python infer.py \
    --model_path runs/detect/qr_yolo_model/weights/best.pt \
    --input_dir data/QR_DS/test_images \
    --output_file outputs/submission_detection_1.json \
    --stage 1
```

**Stage 2: Detection, Decoding & Classification (Bonus Task)**

This will generate the `submission_decoding_2.json` file for the bonus task.

```bash
python infer.py \
    --model_path runs/detect/qr_yolo_model/weights/best.pt \
    --input_dir data/QR_DS/test_images \
    --output_file outputs/submission_decoding_2.json \
    --stage 2 \
    --save_images  # Optional: to save annotated images for review
```