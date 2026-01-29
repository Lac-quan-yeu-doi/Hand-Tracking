import sys

sys.path.append("D:/University/projectS/hand_tracking")
print("****************** ADDED ROOT DIRECTORY ******************")

import cv2
import time
import numpy as np
from typing import List, Tuple

# Import your detector (adjust path if needed)
from src.deploy.core.hand_detector import HandDetector, HandResult


import cv2
from src.deploy.core.hand_detector import HandDetector
from src.deploy.core.finger_tracker import FingerTracker, classify_swipe

# MediaPipe landmark index
MIDDLE_FINGER_TIP = 12


def main():
    detector = HandDetector(
        model_path="../deploy/models/hand_landmarker.task",
        num_hands=1
    )

    tracker = FingerTracker(min_speed=2.5)

    cap = cv2.VideoCapture(0)

    print("Swipe with your MIDDLE finger. ESC to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)

        result = detector.detect(frame)

        if result and result.num_hands > 0:
            hand = result.hand_landmarks[0]
            tip = hand[MIDDLE_FINGER_TIP]

            # normalized coordinates
            x, y = tip.x, tip.y

            velocity = tracker.update(x, y, result.timestamp_ms)

            if velocity:
                vx, vy, speed = velocity
                direction = classify_swipe(vx, vy)

                if direction:
                    print(f"Swipe: {direction} - speed: {speed}")

                    # visualize on screen
                    cv2.putText(
                        frame,
                        f"Swipe: {direction}",
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 0),
                        3
                    )

            frame = detector.draw_landmarks(frame, result)

        cv2.imshow("Swipe Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    detector.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
