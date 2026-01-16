import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------
# Only need to do this once
BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

# Download model from: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
# (lite version is fastest → hand_landmarker.task)
model_path = "model/hand_landmarker.task"   # put it in your project folder

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,   # or IMAGE / LIVE_STREAM
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options( )
# -------------------------------

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)  # mirror - more natural
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Timestamp in milliseconds (important for VIDEO mode)
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

    detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

    # Draw landmarks (you need to write your own drawing function)
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    cv2.imshow("Hand Tracking - New API", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()