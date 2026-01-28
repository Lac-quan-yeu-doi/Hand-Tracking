"""
data_preprocessor.py

Processes your fingers dataset folder structure:
  dataset_root/
  ├── train/
  │   ├── 0/   *.png *.jpg ...
  │   ├── 1/
  │   ...
  │   └── 5/
  ├── test/
  └── val/

For each image:
- load
- run original + augmented versions through HandDetector
- save flattened landmarks (x,y,z × 21) + label to CSV

Requirements:
- frame_detection.py in the same directory
- mediapipe, opencv-python, pandas, numpy installed
"""
import sys
sys.path.append('D:/University/projectS/hand_tracking')

import os
import cv2
import numpy as np
import pandas as pd
import math
from typing import List, Optional
from tqdm import tqdm

from src.deploy.core.frame_detector import HandDetector, HandResult

# ── Configuration ────────────────────────────────────────────────
DATASET_ROOT    = "dataset/dataset-finger-count-0to5"
MODEL_PATH      = "D:/University/projectS/hand_tracking/src/deploy/models/hand_landmarker.task"

# Augmentation settings (moderate to avoid too much distortion)
ROTATION_DEGREES = [-25, -15, -10, 10, 15, 25]
ROTATION_DEGREES = []
SCALE_FACTORS    = [0.92, 1.08]
SHEAR_DEGREES    = [-8, 8]

# Detection settings
MIN_CONFIDENCE   = 0.55           # slightly higher than default to filter noisy detections


def get_affine_transform(scale: float = 1.0, shear_deg: float = 0.0) -> np.ndarray:
    """Return 2×3 affine matrix for scale + shear (around center)"""
    return np.float32([
        [scale, math.tan(math.radians(shear_deg)), 0],
        [0,     scale,                            0]
    ])


def augment_image(img: np.ndarray) -> List[np.ndarray]:
    """Generate list of augmented versions (including original)"""
    variants = [img]  # original is always included

    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    # Rotations
    for angle in ROTATION_DEGREES:
        M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        aug = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        variants.append(aug)

    # Scale + shear combinations
    for scale in SCALE_FACTORS:
        for shear in SHEAR_DEGREES:
            M = get_affine_transform(scale, shear)
            aug = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            variants.append(aug)

    return variants

from tqdm import tqdm

def extract_flat_landmarks(result: Optional[HandResult]) -> Optional[List[float]]:
    """Return flattened [x,y,z, x,y,z, ...] for first hand or None if invalid"""
    if not result:
        return None
    if result.num_hands != 1:
        return None  # we want clean single-hand images

    landmarks = result.hand_landmarks[0]  # first (and only) hand

    flat = []
    for lm in landmarks:
        flat.extend([lm.x, lm.y, lm.z])

    return flat


def main():
    detector = HandDetector(
        model_path=MODEL_PATH,
        num_hands=1,                    # force single hand assumption
        min_detection_confidence=MIN_CONFIDENCE,
        min_tracking_confidence=MIN_CONFIDENCE,
        flip_horizontal=False           # dataset images usually not mirrored
    )

    columns = ["label"] + [f"lm{i}_{c}" for i in range(21) for c in ["x", "y", "z"]]

    for split in ["train", "test", "val"]:
        all_rows = []
        split_dir = os.path.join(DATASET_ROOT, split)
        if not os.path.isdir(split_dir):
            print(f"Directory not found: {split_dir} → skipping")
            continue

        for label_str in ["0", "1", "2", "3", "4", "5"]:
            class_dir = os.path.join(split_dir, label_str)
            if not os.path.isdir(class_dir):
                continue

            label = int(label_str)
            files = [f for f in os.listdir(class_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            print(f"{split}/{label} → {len(files)} images")

            for fname in tqdm(files, desc=f"Process {split}/{label_str}", leave=False):
                path = os.path.join(class_dir, fname)
                img_bgr = cv2.imread(path)
                if img_bgr is None:
                    continue

                # Generate augmented versions (includes original)
                variants = augment_image(img_bgr)

                for variant_img in variants:
                    result = detector.detect(variant_img)
                    flat_coords = extract_flat_landmarks(result)

                    if flat_coords and len(flat_coords) == 21 * 3:
                        all_rows.append([label] + flat_coords)
        output_csv = f"dataset/{split}.csv"
        df = pd.DataFrame(all_rows, columns=columns)
        df.to_csv(output_csv, index=False, float_format="%.6f")
        print(f"\nDone. Saved {len(df):,} rows to {output_csv}")
        print(f"Class distribution:\n{df['label'].value_counts().sort_index()}")
    detector.close()

    if not all_rows:
        print("No valid hand landmarks extracted.")
        return

    


if __name__ == "__main__":
    main()