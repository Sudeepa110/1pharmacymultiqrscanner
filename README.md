# Multi-QR Code Recognition Hackathon Submission

This repository contains the source code for the Multi-QR Code Recognition Hackathon. The solution uses a YOLOv8 model for QR code detection and OpenCV's QRCodeDetector for decoding QR codes, including classification of QR types (batch, manufacturer, distributor, regulator).

## 📁 Repository Structure

```
multiqr-hackathon/
│
├── README.md                              # Setup & usage instructions
├── requirements.txt                       # Python dependencies
├── train.py                              # Model training script
├── infer.py                              # Stage 1: Detection inference
├── infer2.py                             # Stage 2: Decoding inference
│
├── data/
│   └── demo_images/                      # Demo image set
│
├── outputs/
│   ├── submission_detection_1.json       # Stage 1 output file
│   └── submission_decoding_2.json        # Stage 2 output file
│
└── src/
    ├── models/                           # Model weights and outputs
    ├── datasets/                         # Dataset files
    ├── utils/                            # Utility functions
    └── __init__.py
```

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd multiqr-hackathon
```

### 2. Environment Setup

It is recommended to use a Python virtual environment.

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/Mac
source venv/bin/activate

# On Windows
venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt
```

### 3. Data Preparation

1. Place your dataset folder (`QR_DS`) inside the `src/datasets/` directory.

2. Update the `data.yaml` file inside the `src/datasets/` directory with the correct paths to your training and validation sets.

**Example `data.yaml`:**

```yaml
# Path to training images
train: /path/to/your/project/src/datasets/QR_DS/train_images/images

# Path to validation images
val: /path/to/your/project/src/datasets/QR_DS/valid_images/images

# Path to test images
test: /path/to/your/project/src/datasets/QR_DS/valid_images/images

# Number of classes
nc: 1

# Class names
names: ['qr_codes']
```

> **Note:** Replace `/path/to/your/project/` with your actual project directory path.

## ⚙️ Configuration

All scripts have configurable variables at the top that you can modify to match your local setup.

### Training Configuration (`train.py`)

```python
DATA_YAML = "src/datasets/data.yaml"    # Path to dataset config
EPOCHS = 100                             # Number of training epochs
BATCH_SIZE = 16                          # Batch size
IMG_SIZE = 640                           # Image size for training
PATIENCE = 20                            # Early stopping patience
PRETRAINED_MODEL = "yolov8n.pt"         # Pretrained YOLOv8 model
PROJECT_NAME = "src/models"              # Directory to save trained weights
MODEL_NAME = "qr_detector"               # Subfolder for this model
```

### Stage 1 Configuration (`infer.py`)

```python
MODEL_PATH = r"D:\multiqr-hackaton\src\models\qr_detector\weights\best.pt"
INPUT_DIR = r"D:\multiqr-hackaton\src\datasets\QR_DS\test_images"
OUTPUT_DIR = r"D:\multiqr-hackaton\outputs"
SAVE_IMAGES = True
```

### Stage 2 Configuration (`infer2.py`)

```python
MODEL_PATH = r"D:\multiqr-hackaton\src\models\qr_detector\weights\best.pt"
INPUT_DIR = r"D:\multiqr-hackaton\src\datasets\QR_DS\test_images"
OUTPUT_DIR = r"D:\multiqr-hackaton\outputs"
SAVE_IMAGES = True
```

## 🎯 Usage

### Training the Model

To train the YOLOv8 model for QR code detection:

```bash
python train.py
```

The trained model weights (`best.pt`) will be saved in `src/models/qr_detector/weights/`.

### Stage 1: QR Code Detection

This stage detects QR codes in images and generates bounding box predictions.

1. **Update Configuration:**
<<<<<<< HEAD
   - Copy the path of `best.pt` from `src/models/qr_detector/weights/`
   - Copy the path of `test_images` from `src/datasets/QR_DS/`
   - Copy the path of `outputs folder` from `outputs`
   - Paste these paths into `MODEL_PATH`, `INPUT_DIR` and `OUTPUT_DIR` in `infer.py`
=======
   - Copy the path to `best.pt` from `src/models/qr_detector/weights/`
   - Copy the path to `test_images` from `src/datasets/QR_DS/`
   - Paste these paths into `MODEL_PATH` and `INPUT_DIR` in `infer.py`
>>>>>>> e6bed7ad794a8216eab5fc5e88f018a7aa4460c6

2. **Run Detection:**

```bash
python infer.py
```

3. **Output:**
   - Annotated images: `src/models/predict_stage1_annotated/`
   - JSON results: `outputs/submission_detection_1.json`
<<<<<<< HEAD

### Stage 2: QR Code Decoding

This stage decodes the detected QR codes and classifies them by type.

1. **Update Configuration:**
   - Copy the path of `best.pt` from `src/models/qr_detector/weights/`
   - Copy the path of `test_images` from `src/datasets/QR_DS/`
   - Copy the path of `outputs folder` from `multiqr-hackathon/outputs`
   - Paste these paths into `MODEL_PATH`, `INPUT_DIR` and `OUTPUT_DIR`in `infer2.py`

2. **Run Decoding:**

```bash
python infer2.py
```

3. **Output:**
   - Annotated images: `src/models/predict_stage2_annotated/`
   - JSON results: `outputs/submission_decoding_2.json`

## 📊 Output Format

### Stage 1 (Detection)

```json
{
  "image_name.jpg": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95
    }
  ]
}
```

### Stage 2 (Decoding)

```json
{
  "image_name.jpg": [
    {
      "bbox": [x1, y1, x2, y2],
      "decoded_text": "QR_CODE_CONTENT",
      "qr_type": "batch"
    }
  ]
}
```

## 📋 Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy

See `requirements.txt` for the complete list of dependencies.

📝 Notes

Ensure all paths in configuration files use absolute paths or are relative to the project root.
The data.yaml file must be updated before training.
Model weights are saved automatically during training with early stopping enabled.
=======

### Stage 2: QR Code Decoding

This stage decodes the detected QR codes and classifies them by type.

1. **Update Configuration:**
   - Copy the path to `best.pt` from `src/models/qr_detector/weights/`
   - Copy the path to `test_images` from `src/datasets/QR_DS/`
   - Paste these paths into `MODEL_PATH` and `INPUT_DIR` in `infer2.py`

2. **Run Decoding:**

```bash
python infer2.py
```

3. **Output:**
   - Annotated images: `src/models/predict_stage2_annotated/`
   - JSON results: `outputs/submission_decoding_2.json`

## 📊 Output Format

### Stage 1 (Detection)

```json
{
  "image_name.jpg": [
    {
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95
    }
  ]
}
```

### Stage 2 (Decoding)

```json
{
  "image_name.jpg": [
    {
      "bbox": [x1, y1, x2, y2],
      "decoded_text": "QR_CODE_CONTENT",
      "qr_type": "batch"
    }
  ]
}
```

## 📋 Requirements

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy

See `requirements.txt` for the complete list of dependencies.

>>>>>>> e6bed7ad794a8216eab5fc5e88f018a7aa4460c6
