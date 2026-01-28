"""
data_preprocessor.py

This script:
1. Traverses the dataset folders (train/test/val > 0-5)
2. For each image:
   - Loads it
   - Applies augmentations (rotations + affine)
   - Runs HandDetector to extract landmarks
   - Saves original + augmented features to CSV

Assumes:
- frame_detection.py in same folder
- models/hand_landmarker.task exists
- Dataset root is './dataset/' (adjust DATASET_ROOT)
- cv2, mediapipe installed locally

Output: features.csv with columns:
- label: int (0-5)
- lm0_x, lm0_y, lm0_z, lm1_x, ... lm20_z  (21 landmarks * 3 coords)
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import List
import math

# Import your detector
from frame_detection import HandDetector, HandResult

# Config
DATASET_ROOT = "."  # Adjust if needed, e.g., "./fingers_dataset"
MODEL_PATH = "models/hand_landmarker.task"
OUTPUT_CSV = "hand_landmarks_features.csv"

# Augmentation params
ROTATIONS = [-30, -15, 15, 30]  # degrees
AFFINE_SCALES = [0.9, 1.1]      # slight zoom in/out
AFFINE_SHEARS = [5, -5]        # degrees shear


def apply_rotation(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by angle degrees"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)


def apply_affine(image: np.ndarray, scale: float = 1.0, shear: float = 0.0) -> np.ndarray:
    """Apply scale and shear affine transform"""
    h, w = image.shape[:2]
    affine_matrix = np.float32([
        [scale, math.tan(math.radians(shear)), 0],
        [0, scale, 0]
    ])
    return cv2.warpAffine(image, affine_matrix, (w, h), flags=cv2.INTER_LINEAR)


def extract_landmarks(detector: HandDetector, image: np.ndarray) -> List[float]:
    """
    Run detection, extract flattened [x,y,z] * 21 for first hand (if detected)
    Returns empty list if no hand or error
    """
    result = detector.detect(image)
    if not result or not result.hand_landmarks:
        return []

    # Take first hand (assume one main hand per image)
    landmarks = result.hand_landmarks[0]  # List[NormalizedLandmark]

    flat_coords = []
    for lm in landmarks:
        flat_coords.extend([lm.x, lm.y, lm.z])

    return flat_coords


def main():
    detector = HandDetector(
        model_path=MODEL_PATH,
        num_hands=1,  # Assume one hand per image
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        flip_horizontal=False  # Dataset likely not flipped
    )

    # Prepare DataFrame
    columns = ['label'] + [f'lm{i}_{coord}' for i in range(21) for coord in ['x', 'y', 'z']]
    data_rows = []

    # Traverse dataset
    for split in ['train', 'test', 'val']:
        split_path = os.path.join(DATASET_ROOT, split)
        if not os.path.exists(split_path):
            print(f"Skipping {split} - path not found")
            continue

        for label_str in ['0', '1', '2', '3', '4', '5']:
            label = int(label_str)
            folder_path = os.path.join(split_path, label_str)
            if not os.path.exists(folder_path):
                continue

            images = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            for img_name in images:
                img_path = os.path.join(folder_path, img_name)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                # Original
                coords = extract_landmarks(detector, image)
                if coords:
                    data_rows.append([label] + coords)

                # Augmentations
                # Rotations
                for angle in ROTATIONS:
                    aug_img = apply_rotation(image, angle)
                    coords = extract_landmarks(detector, aug_img)
                    if coords:
                        data_rows.append([label] + coords)

                # Affines (scale + shear combos)
                for scale in AFFINE_SCALES:
                    for shear in AFFINE_SHEARS:
                        aug_img = apply_affine(image, scale=scale, shear=shear)
                        coords = extract_landmarks(detector, aug_img)
                        if coords:
                            data_rows.append([label] + coords)

            print(f"Processed {split}/{label_str} - {len(images)} images")

    # Save to CSV
    df = pd.DataFrame(data_rows, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(df)} samples to {OUTPUT_CSV}")

    detector.close()


if __name__ == "__main__":
    main()