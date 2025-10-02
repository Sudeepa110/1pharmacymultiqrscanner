import os
import json
import cv2
from ultralytics import YOLO
from pyzbar.pyzbar import decode as pyzbar_decode
import argparse
from tqdm import tqdm

def classify_qr(value):
    """
    Classifies the QR code type based on its decoded value.
    Customize these rules based on your dataset patterns.
    """
    if value.startswith(("B", "5a0S")):
        return "batch"
    elif value.startswith("MFR"):
        return "manufacturer"
    elif value.startswith("D"):
        return "distributor"
    elif value.startswith("R"):
        return "regulator"
    else:
        return "unknown"

def run_inference(model_path, input_dir, output_file, stage, save_images):
    """
    Runs inference on a directory of images to detect, decode, and classify QR codes.
    """
    # Create output directories if they don't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    save_images_dir = "runs/detect/predict_annotated"
    if save_images:
        os.makedirs(save_images_dir, exist_ok=True)

    # Load the trained YOLOv8 model
    model = YOLO(model_path)
    submission = []

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Running inference for Stage {stage} on {len(image_files)} images...")
    
    for img_file in tqdm(image_files, desc="Processing Images"):
        img_path = os.path.join(input_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_file}. Skipping.")
            continue

        # Predict bounding boxes using the YOLO model
        results = model.predict(img_path, save=False, conf=0.25, verbose=False)
        r = results[0]  # Get results for the first image
        qrs = []

        for box in r.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            
            if stage == 1:
                # Stage 1: Detection only
                qrs.append({
                    "bbox": [float(x) for x in box]
                })
            elif stage == 2:
                # Stage 2: Detection, Decoding, and Classification
                crop = img[y1:y2, x1:x2]
                
                qr_value = "unknown"
                qr_type = "unknown"

                # Decode QR code using pyzbar
                decoded_objs = pyzbar_decode(crop)
                if decoded_objs:
                    qr_value = decoded_objs[0].data.decode("utf-8")
                
                # Classify the QR code
                qr_type = classify_qr(qr_value)

                qrs.append({
                    "bbox": [float(x) for x in box],
                    "value": qr_value,
                    "type": qr_type
                })

            # Draw annotations on the image if requested
            if save_images:
                label = f"{qr_value} ({qr_type})" if stage == 2 else "qr_code"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        submission.append({"image_id": img_file, "qrs": qrs})

        if save_images:
            save_path = os.path.join(save_images_dir, img_file)
            cv2.imwrite(save_path, img)

    # Save the final submission JSON
    with open(output_file, "w") as f:
        json.dump(submission, f, indent=2)

    print(f"\n✅ Submission JSON saved to: {output_file}")
    if save_images:
        print(f"✅ Annotated images saved to: {save_images_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for Multi-QR Hackathon.")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained YOLOv8 model (best.pt).')
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing test images.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the output submission JSON file.')
    parser.add_argument('--stage', type=int, choices=[1, 2], required=True, help='Specify the submission stage (1 for detection, 2 for decoding).')
    parser.add_argument('--save_images', action='store_true', help='Set this flag to save images with annotations.')

    args = parser.parse_args()

    run_inference(
        model_path=args.model_path,
        input_dir=args.input_dir,
        output_file=args.output_file,
        stage=args.stage,
        save_images=args.save_images
    )