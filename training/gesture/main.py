import cv2
import os
import time
import math
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from resnext import resnext101


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAMPLE_DURATION = 32   # MUST match pretrained model
SAMPLE_SIZE = 112
NUM_CLASSES = 27
CONF_THRESHOLD = 0.1


JESTER_LABELS = [
    "Swiping Left", "Swiping Right", "Swiping Down", "Swiping Up",
    "Sliding Two Fingers Left", "Sliding Two Fingers Right",
    "Sliding Two Fingers Down", "Sliding Two Fingers Up",
    "Pushing Hand Away", "Pulling Hand In",
    "Pushing Two Fingers Away", "Pulling Two Fingers In",
    "Rolling Hand Forward", "Rolling Hand Backward",
    "Turning Hand Clockwise", "Turning Hand Counterclockwise",
    "Zooming In With Full Hand", "Zooming Out With Full Hand",
    "Zooming In With Two Fingers", "Zooming Out With Two Fingers",
    "Thumb Up", "Thumb Down", "Shaking Hand", "Stop Sign",
    "Drumming Fingers", "No gesture", "Doing other things"
]

print(len(JESTER_LABELS), "gesture classes loaded")

# ─────────────────────────────────────────────────────────────
# HAND ROI UTILITIES
# ─────────────────────────────────────────────────────────────
def crop_hand(frame, landmarks, margin=0.3):
    h, w, _ = frame.shape
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]

    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))

    dx = int((x2 - x1) * margin)
    dy = int((y2 - y1) * margin)

    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(w, x2 + dx)
    y2 = min(h, y2 + dy)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def preprocess_frame(frame):
    frame = cv2.resize(frame, (SAMPLE_SIZE, SAMPLE_SIZE))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    frame = (frame - 0.5) / 0.5  # [-1, 1]
    return frame


def build_clip(frames):
    clip = np.stack(frames, axis=0)          # (T,H,W,3)
    clip = clip.transpose(3, 0, 1, 2)        # (3,T,H,W)
    clip = np.expand_dims(clip, axis=0)      # (1,3,T,H,W)
    return torch.from_numpy(clip).float()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    # ─── Load ResNeXt ────────────────────────────────────────
    model = resnext101(
        sample_size=SAMPLE_SIZE,
        sample_duration=SAMPLE_DURATION,
        num_classes=NUM_CLASSES
    )

    # Load pretrained weights -> state_dict keys start with 'module.' but model structure doesn't need need-> fix this
    ckpt = torch.load("../pretrain/models/jester_resnext_101_RGB_32.pth", map_location="cpu")

    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        new_key = k.replace("module.", "", 1)
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict, strict=True)

    model.eval().to(DEVICE)
    print("[INFO] ResNeXt-101 loaded")

    # ─── MediaPipe HandLandmarker ─────────────────────────────
    BaseOptions = python.BaseOptions
    HandLandmarker = vision.HandLandmarker
    HandLandmarkerOptions = vision.HandLandmarkerOptions
    RunningMode = vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="../../src/models/hand_landmarker.task"),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    landmarker = HandLandmarker.create_from_options(options)

    # ─── Video ───────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    frame_buffer = []

    print("[INFO] Running demo — press ESC to quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            roi = crop_hand(frame, landmarks)

            if roi is not None:
                roi = preprocess_frame(roi)
                frame_buffer.append(roi)

                if len(frame_buffer) > SAMPLE_DURATION:
                    frame_buffer.pop(0)

                for lm in landmarks:
                    cx = int(lm.x * frame.shape[1])
                    cy = int(lm.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        gesture_text = "Collecting..."
        color = (0, 0, 255)

        if len(frame_buffer) == SAMPLE_DURATION:
            clip = build_clip(frame_buffer).to(DEVICE)

            with torch.no_grad():
                logits = model(clip)
                probs = F.softmax(logits, dim=1)
                idx = torch.argmax(probs, dim=1).item()
                print("Probs:", probs.cpu().numpy())
                conf = probs[0, idx].item()

            if conf > CONF_THRESHOLD:
                gesture_text = f"{JESTER_LABELS[idx]} ({conf:.2f})"
                color = (0, 255, 0)
            else:
                gesture_text = "No gesture"

        cv2.putText(frame, gesture_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Jester Gesture Recognition (ResNeXt)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
