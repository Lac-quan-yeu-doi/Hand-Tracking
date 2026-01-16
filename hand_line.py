import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─── Model & landmarker setup (your existing code) ───
model_path = "model/hand_landmarker.task"  # or "model/hand_landmarker.task"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

# ─── Hand connections (same as classic MediaPipe Hands) ───
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (0, 17),                     # pinky - palm
    (17, 18), (18, 19), (19, 20)           # pinky
]

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # ─── Draw hand skeleton ──────────────────────────────────────
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:  # list of lists (one per hand)
            h, w, _ = frame.shape

            # 1. Draw connections (lines) first → behind the dots
            for connection in HAND_CONNECTIONS:
                start_idx, end_idx = connection
                if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                    start = hand_landmarks[start_idx]
                    end = hand_landmarks[end_idx]

                    start_x = int(start.x * w)
                    start_y = int(start.y * h)
                    end_x   = int(end.x   * w)
                    end_y   = int(end.y   * h)

                    cv2.line(frame, 
                             (start_x, start_y), 
                             (end_x,   end_y), 
                             (255, 0, 200),   # nice purple-pink
                             thickness=2)

            # 2. Draw smaller dots (landmarks) on top
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)

                # Small, clean dots - you can change radius & color
                cv2.circle(frame, (x, y), 
                          radius=2,               # ← smaller than before (was 5)
                          color=(0, 255, 0),      # green
                          thickness=-1)           # filled

    cv2.imshow("Hand Skeleton - Press ESC", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()