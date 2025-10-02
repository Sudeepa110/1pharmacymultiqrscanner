import os
import yaml
from ultralytics import YOLO
import argparse

def train_model(data_yaml, epochs, batch_size, img_size, patience, model_name, project_name):
    """
    Trains the YOLOv8 model on the QR code dataset.
    """
    # Load a pre-trained YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    print("Starting model training...")
    # Train the model
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=patience,  
        name=model_name,
        project=project_name,
        exist_ok=True 
    )
    print(f"✅ Training complete. Model saved in '{project_name}/{model_name}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 model for QR code detection.")
    parser.add_argument('--data_yaml', type=str, required=True, help='Path to the data.yaml configuration file.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for training.')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience (epochs).') # <-- ADDED THIS ARGUMENT
    parser.add_argument('--model_name', type=str, default='qr_yolo_model', help='Name for the trained model run.')
    parser.add_argument('--project_name', type=str, default='runs/detect', help='Directory to save training runs.')

    args = parser.parse_args()

    train_model(
        data_yaml=args.data_yaml,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        patience=args.patience,
        model_name=args.model_name,
        project_name=args.project_name
    )