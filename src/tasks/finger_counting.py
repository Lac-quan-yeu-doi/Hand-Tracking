import sys

sys.path.append("D:/University/projectS/hand_tracking")
print("****************** ADDED ROOT DIRECTORY ******************")


import cv2
import time
import numpy as np
import joblib
from typing import List, Tuple

# Import your detector (adjust path if needed)
from src.deploy.core.hand_detector import HandDetector, HandResult

# ────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ────────────────────────────────────────────────────────────────

MODEL_PATH = "../deploy/models/hand_landmarker.task"
ML_MODEL_PATH = "../training/finger_counting/models/KNN.joblib"  # ← your saved model
ML_SCALER_PATH = (
    "../training/finger_counting/models/scaler.joblib"  # ← your saved scaler
)

USE_ML = True  # ← change to False to use old heuristic
ML_NUM_HANDS_EXPECT = 2  # most finger-counting assumes 1 main hand

# Landmark indices (same as before)
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18
THUMB_IP = 3


# ────────────────────────────────────────────────────────────────
#  HEURISTIC METHOD (your original)
# ────────────────────────────────────────────────────────────────


def is_finger_up(
    landmarks,
    tip_idx: int,
    pip_idx: int,
    is_thumb: bool = False,
    handedness: str = "Right",
    threshold: float = 0.02,
) -> bool:
    tip = landmarks[tip_idx]
    ref = landmarks[pip_idx]

    if is_thumb:
        if handedness == "Right":
            return tip.x > ref.x + threshold
        else:
            return tip.x < ref.x - threshold
    else:
        return tip.y < ref.y - threshold


def count_raised_fingers_heuristic(result: HandResult) -> Tuple[int, List[str]]:
    if not result or not result.hand_landmarks:
        return 0, []

    total = 0
    names = []
    handedness_list = result.get_handedness_labels()

    for i in range(result.num_hands):
        lm = result.hand_landmarks[i]
        side = handedness_list[i] if i < len(handedness_list) else "Unknown"

        # Thumb
        if is_finger_up(lm, THUMB_TIP, THUMB_IP, True, side):
            total += 1
            names.append(f"Thumb ({side})")

        # Others
        for tip, pip, name in [
            (INDEX_TIP, INDEX_PIP, "Index"),
            (MIDDLE_TIP, MIDDLE_PIP, "Middle"),
            (RING_TIP, RING_PIP, "Ring"),
            (PINKY_TIP, PINKY_PIP, "Pinky"),
        ]:
            if is_finger_up(lm, tip, pip, False):
                total += 1
                names.append(f"{name} ({side})")

    return total, names


# ────────────────────────────────────────────────────────────────
#  ML METHOD
# ────────────────────────────────────────────────────────────────


def extract_features(result: HandResult) -> np.ndarray | None:
    """
    Extract the same 63 features your model was trained on.
    Returns None if invalid (no hand, multiple hands, etc.)
    """
    if not result or result.num_hands != ML_NUM_HANDS_EXPECT:
        return None

    # Take the first hand (or the one with highest confidence if you want to improve)
    landmarks = result.hand_landmarks[0]

    features = []
    for lm in landmarks:
        features.extend([lm.x, lm.y, lm.z])

    return np.array(features).reshape(1, -1)  # shape (1, 63)


def count_raised_fingers_ml(result: HandResult, model, scaler) -> Tuple[int, List[str]]:
    features = extract_features(result)
    if features is None:
        return 0, []

    features_scaled = scaler.transform(features)
    pred = model.predict(features_scaled)[0]  # integer 0–5
    prob = model.predict_proba(features_scaled)[0]  # optional confidence

    # For display: we show the predicted number
    # You can also map back to finger names if you trained per-finger classifiers
    count = int(pred)
    # Simple name approximation — improve this if you want
    finger_names = {
        0: [],
        1: ["Index"],
        2: ["Index", "Middle"],
        3: ["Index", "Middle", "Ring"],
        4: ["Index", "Middle", "Ring", "Pinky"],
        5: ["Thumb", "Index", "Middle", "Ring", "Pinky"],
    }.get(count, ["?"])

    names_with_side = [
        f"{name} ({result.get_handedness_labels()[0]})" for name in finger_names
    ]

    return count, names_with_side


# ────────────────────────────────────────────────────────────────
#  MAIN LOOP
# ────────────────────────────────────────────────────────────────


def main():
    detector = HandDetector(
        model_path=MODEL_PATH,
        num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        flip_horizontal=True,
    )

    # Load ML model & scaler (only if using ML)
    model = None
    scaler = None
    global USE_ML
    if USE_ML:
        try:
            model = joblib.load(ML_MODEL_PATH)
            scaler = joblib.load(ML_SCALER_PATH)
            print("ML model and scaler loaded successfully")
        except Exception as e:
            print(f"Error loading ML model: {e}")
            print("Falling back to heuristic method")
            USE_ML = False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    prev_time = time.time()

    print("Finger counting started. Press ESC to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        result = detector.detect(frame)

        display_frame = frame.copy()

        count = 0
        fingers_list = []
        method_used = ""

        if result:
            detector.draw_landmarks(
                display_frame,
                result,
                with_skeleton=True,
                landmark_color=(0, 255, 120),
                connection_color=(220, 100, 255),
                landmark_radius=4,
                connection_thickness=2,
            )

            if USE_ML and model is not None and scaler is not None:
                count, fingers_list = count_raised_fingers_ml(result, model, scaler)
                method_used = "ML"
            else:
                count, fingers_list = count_raised_fingers_heuristic(result)
                method_used = "Heuristic"

            handed_str = ", ".join(result.get_handedness_labels()) or "Unknown"
            finger_str = ", ".join(fingers_list) if fingers_list else "None"

            cv2.putText(
                display_frame,
                f"Fingers up: {count}  ({method_used})",
                (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 100),
                3,
            )
            cv2.putText(
                display_frame,
                f"→ {finger_str}",
                (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 220, 255),
                2,
            )
            cv2.putText(
                display_frame,
                f"Hands: {handed_str}",
                (30, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (180, 180, 255),
                2,
            )

        # FPS
        now = time.time()
        fps = 1 / (now - prev_time + 1e-8)
        prev_time = now
        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f}",
            (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 180, 255),
            2,
        )

        cv2.imshow("Finger Counting", display_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()
