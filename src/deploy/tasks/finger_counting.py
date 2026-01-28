"""
finger_counting.py

Simple finger counting using MediaPipe hand landmarks.
Imports HandDetector from frame_detection.py
"""

import cv2
import time
from typing import List, Tuple

from core.frame_detector import HandDetector, HandResult

# Landmark indices we care about
THUMB_TIP   = 4
INDEX_TIP   = 8
MIDDLE_TIP  = 12
RING_TIP    = 16
PINKY_TIP   = 20

# PIP joints (for non-thumb fingers)
INDEX_PIP   = 6
MIDDLE_PIP  = 10
RING_PIP    = 14
PINKY_PIP   = 18

# For thumb we usually compare to IP joint (3)
THUMB_IP    = 3


def is_finger_up(
    landmarks,
    tip_idx: int,
    pip_idx: int,
    is_thumb: bool = False,
    handedness: str = "Right",
    threshold: float = 0.02
) -> bool:
    """
    Decide if a finger is raised.
    For thumb: compares x-position (horizontal extension)
    For other fingers: compares y-position (vertical extension)
    """
    tip = landmarks[tip_idx]
    ref = landmarks[pip_idx]   # reference point

    if is_thumb:
        # Thumb logic depends on handedness
        if handedness == "Right":
            return tip.x > ref.x + threshold
        else:  # Left hand
            return tip.x < ref.x - threshold
    else:
        # Other fingers: tip should be significantly higher (smaller y)
        return tip.y < ref.y - threshold


def count_raised_fingers(result: HandResult) -> Tuple[int, List[str]]:
    """
    Count how many fingers are raised for each hand.
    Returns total count and list of raised finger names (across all hands).
    """
    if not result or not result.hand_landmarks:
        return 0, []

    total_count = 0
    raised_names = []

    handedness_list = result.get_handedness_labels()

    for hand_idx in range(result.num_hands):
        landmarks = result.hand_landmarks[hand_idx]
        hand_side = handedness_list[hand_idx] if handedness_list else "Unknown"

        # ─── Thumb ────────────────────────────────────────────────
        if is_finger_up(landmarks, THUMB_TIP, THUMB_IP, is_thumb=True, handedness=hand_side):
            total_count += 1
            raised_names.append(f"Thumb ({hand_side})")

        # ─── Index, Middle, Ring, Pinky ───────────────────────────
        for tip, pip, name in [
            (INDEX_TIP,   INDEX_PIP,   "Index"),
            (MIDDLE_TIP,  MIDDLE_PIP,  "Middle"),
            (RING_TIP,    RING_PIP,    "Ring"),
            (PINKY_TIP,   PINKY_PIP,   "Pinky"),
        ]:
            if is_finger_up(landmarks, tip, pip, is_thumb=False):
                total_count += 1
                raised_names.append(f"{name} ({hand_side})")

    return total_count, raised_names


def main():
    MODEL_PATH = "models/hand_landmarker.task"  # adjust if needed

    detector = HandDetector(
        model_path=MODEL_PATH,
        num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        flip_horizontal=True
    )

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

        # Mirror webcam feed (most natural for user)
        frame = cv2.flip(frame, 1)

        result = detector.detect(frame)

        display_frame = frame.copy()

        if result:
            # Draw landmarks + connections
            detector.draw_landmarks(
                display_frame,
                result,
                with_skeleton=True,
                landmark_color=(0, 255, 120),
                connection_color=(220, 100, 255),
                landmark_radius=4,
                connection_thickness=2
            )

            count, fingers_list = count_raised_fingers(result)

            # Status text
            handed_str = ", ".join(result.get_handedness_labels()) or "Unknown"
            finger_str = ", ".join(fingers_list) if fingers_list else "None"

            cv2.putText(display_frame, f"Fingers up: {count}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 100), 3)
            cv2.putText(display_frame, f"→ {finger_str}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 220, 255), 2)
            cv2.putText(display_frame, f"Hands: {handed_str}", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 255), 2)

        # Show FPS
        now = time.time()
        fps = 1 / (now - prev_time + 1e-8)
        prev_time = now
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 180, 255), 2)

        cv2.imshow("Finger Counting", display_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()