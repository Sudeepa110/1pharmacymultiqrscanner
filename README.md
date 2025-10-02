# Multi-QR Code Recognition Hackathon Submission

This repository contains the source code for the Multi-QR Code Recognition Hackathon. The solution uses a YOLOv8 model for QR code detection and OpenCV’s QRCodeDetector for decoding QR codes, including classification of QR types (batch, manufacturer, distributor, regulator).
## Repository Structure

│
├── README.md                # Setup & usage instructions
├── requirements.txt         # Python deps
├── train.py                 #  training script
├── infer.py                 # Must implement inference (input=images → output=JSON)
├── evaluate.py              # (Optional) for self-check with provided GT
│
├── data/                    # (participants don't commit dataset, only placeholder)
│   └── demo_images/         # You’ll provide a small demo set
│
├── outputs/                 
│   ├── submission_detection_1.json   # Required output file (Stage 1)
│   └── submission_decoding_2.json    # Required output file (Stage 2, bonus)
│
└── src/                     # Their actual model code, utils, data loaders, etc.
    ├── models/
    ├── datasets/
    ├── utils/
    └── __init__.py


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

# On Linux/Mac
source venv/bin/activate
# On Windows
venv\Scripts\activate


# Install Python packages
pip install -r requirements.txt
```

### 3. Data Preparation
3. Data Preparation

1. Place your dataset folder (QR_DS) inside the src/datasets/ directory.

2.  Update the`data.yaml` file inside the `datasets/` directory with the correct paths to your training and validation sets. **You must update the paths below to match your local setup.**

    ```yaml
    
    # data/data.yaml
    # Copy the path of train,valid,test images and paste in the data.yaml file
    train: /path/to/your/project/data/QR_DS/train_images/images
    val: /path/to/your/project/data/QR_DS/valid_images/images
    test:/path/to/your/project/data/QR_DS/valid_images/images

    # Number of classes
    nc: 1

    # Class names
    names: ['qr_codes']
    ```

## Customizing Scripts

All scripts (train.py, infer_stage1.py, infer_stage2.py) have variables at the top that you can modify to match your local setup or experiment parameters.

  ## Training (train.py) 
        ```yaml
            DATA_YAML = "src/datasets/data.yaml"  # Path to dataset config
            EPOCHS = 100                           # Number of training epochs
            BATCH_SIZE = 16                        # Batch size
            IMG_SIZE = 640                          # Image size for training
            PATIENCE = 20                           # Early stopping patience
            PRETRAINED_MODEL = "yolov8n.pt"        # Pretrained YOLOv8 model
            PROJECT_NAME = "src/models"            # Directory to save trained weights
            MODEL_NAME = "qr_detector"             # Subfolder for this model
        ```
## Training

To train the model from scratch, run the `train.py` script. The trained model weights (`best.pt`) will be saved in the `runs/detect/qr_yolo_model/` directory.

```bash
python train.py 

```
  
  
  
   ## Stage 1 (infer.py) #DETECTION
        Copy the path of
                            1. best.pt by going to src\models\qr_detector\weights\
                            2. test_images by going to src\datasets\QR_DS\
                   
                   and paste it in the MODEL_PATH and INPUT_DIR resectively in the file infer.py

                    ```yaml
                            MODEL_PATH = r"D:\multiqr-hackaton\src\models\qr_detector\weights\best.pt"
                            INPUT_DIR = r"D:\multiqr-hackaton\src\datasets\QR_DS\test_images"
                            OUTPUT_DIR = r"D:\multiqr-hackaton\outputs"
                            SAVE_IMAGES = True  
                    ```

            Then Run python infer.py to get the Detection Of Scanner. The test_images  qr detected will be in the folder 
            src/models/predict_stage1_annotated and the json file will be in the Outputs folder named submission_detection_1.json




   ## Stage 2 (infer2.py) #DECODING
           Copy the path of       
                            1. best.pt by going to src\models\qr_detector\weights\
                            2. test_images by going to src\datasets\QR_DS\
                   
                   and paste it in the MODEL_PATH and INPUT_DIR resectively in the file infer.py

        ```yaml
                MODEL_PATH = r"D:\multiqr-hackaton\src\models\qr_detector\weights\best.pt"
                INPUT_DIR = r"D:\multiqr-hackaton\src\datasets\QR_DS\test_images"
                OUTPUT_DIR = r"D:\multiqr-hackaton\outputs"
                SAVE_IMAGES = True
        ```

     Then Run python infer2.py to get the Decoding Of Scanner. The test_images  qr decoded will be in the folder 
            src/models/predict_stage2_annotated and the json file will be in the Outputs folder named submission_detection_2.json
```