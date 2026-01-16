import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional, Tuple
import time

class HandResult:
    """
    Result for hand detection in a frame. Coordinates are NORMALIZED [0.0, 1.0].
    """
    def __init__(
        self,
        hand_landmarks,              # List[List[NormalizedLandmark]]
        handedness,                  # List[List[Category]]
        num_hands: int,
        timestamp_ms: int,
        frame_shape: Tuple[int, int, int]
    ):
        self.hand_landmarks = hand_landmarks
        self.handedness = handedness
        self.num_hands = num_hands
        self.timestamp_ms = timestamp_ms
        self.frame_shape = frame_shape  # (height, width, channels)

    def get_handedness_labels(self) -> list[str]:
        """Returns list of handedness strings ('Left'/'Right') for each hand"""
        if not self.handedness:
            return ["Unknown"] * self.num_hands
        return [h[0].category_name for h in self.handedness]

class HandDetector:
    """
    Responsible ONLY for MediaPipe Hand Landmarker detection.
    
    Input:  BGR frame (from camera or any source)
    Output: HandResult object or None if no hands detected
    
    Does NOT:
    - draw anything
    - handle mouse control
    - perform gesture recognition
    """

    # Static attributes
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # index
        (5, 9), (9, 10), (10, 11), (11, 12),   # middle
        (9, 13), (13, 14), (14, 15), (15, 16), # ring
        (13, 17), (0, 17),                     # pinky - palm
        (17, 18), (18, 19), (19, 20)           # pinky
    ]
    
    def __init__(
        self,
        model_path: str,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        flip_horizontal: bool = True,
    ):
        self.flip_horizontal = flip_horizontal
        
        # Initialize MediaPipe Hand Landmarker
        self.BaseOptions = python.BaseOptions
        self.HandLandmarker = vision.HandLandmarker
        self.HandLandmarkerOptions = vision.HandLandmarkerOptions
        self.VisionRunningMode = vision.RunningMode

        options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=model_path),
            running_mode=self.VisionRunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.landmarker = self.HandLandmarker.create_from_options(options)
        print(f"HandDetector initialized (max hands: {num_hands})")

    def detect(self, frame: cv2.Mat) -> Optional[HandResult]:
        """
        Detect hands in a BGR frame    
        Args:
            frame: BGR image (numpy array from cv2)
            
        Returns:
            HandResult object or None if no hands found or error
        """
        if frame is None or frame.size == 0:
            return None

        # Convert BGR → RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image object
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # Get current timestamp (for VIDEO tracking)
        timestamp_ms = int(time.time() * 1000)

        # Run detection
        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        # No hands detected
        if not detection_result.hand_landmarks:
            return None

        if self.flip_horizontal:
            # If frame is flipped, flip handedness
            for hand_handedness in detection_result.handedness:
                for category in hand_handedness:
                    if category.category_name == 'Left':
                        category.category_name = 'Right'
                    elif category.category_name == 'Right':
                        category.category_name = 'Left'

        result = [
            detection_result.hand_landmarks,
            detection_result.handedness,
            len(detection_result.hand_landmarks),
            timestamp_ms,
            frame.shape
        ]
                
        return HandResult(
            hand_landmarks=result[0],
            handedness=result[1],
            num_hands=result[2],
            timestamp_ms=result[3],
            frame_shape=result[4]
        )

    def draw_landmarks(
        self,
        frame: cv2.Mat,
        detection_result: Optional[vision.HandLandmarkerResult],
        with_skeleton: bool = True,
        landmark_color: Tuple[int, int, int] = (0, 255, 0),    # green
        landmark_radius: int = 2,
        connection_color: Tuple[int, int, int] = (255, 0, 200), # purple-pink
        connection_thickness: int = 2
    ) -> cv2.Mat:
        """
        Draw hand landmarks and optionally the skeleton connections on the frame.
        
        Args:
            frame: The BGR frame, modified in-place
            detection_result: HandResult from detect()
            with_skeleton: With skeleton or not
            landmark_color: Landmark color (BGR)
            landmark_radius: Landmark dot radius
            connection_color: Skeleton color (BGR)
            connection_thickness: Skeleton line thickness
            
        Returns:
            The modified frame (same object, drawn in-place)
        """
        if detection_result is None or not detection_result.hand_landmarks:
            return frame

        h, w, _ = frame.shape

        for hand_landmarks in detection_result.hand_landmarks:
            # Draw connections (skeleton lines) first - they go behind dots
            if with_skeleton:
                for start_idx, end_idx in self.HAND_CONNECTIONS:
                    if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                        start = hand_landmarks[start_idx]
                        end = hand_landmarks[end_idx]

                        start_x = int(start.x * w)
                        start_y = int(start.y * h)
                        end_x   = int(end.x   * w)
                        end_y   = int(end.y   * h)

                        cv2.line(
                            frame,
                            (start_x, start_y),
                            (end_x, end_y),
                            connection_color,
                            connection_thickness
                        )

            # Draw landmark dots on top
            for lm in hand_landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)

                cv2.circle(
                    frame,
                    (x, y),
                    radius=landmark_radius,
                    color=landmark_color,
                    thickness=-1  # filled circle
                )

        return frame

    def close(self):
        """Release MediaPipe resources"""
        if hasattr(self, 'landmarker') and self.landmarker is not None:
            self.landmarker.close()
            print("HandDetector resources released")

    def __del__(self):
        self.close()

# ──────────────────────────────────────────────────────────────
# Simple test / example usage (standalone)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    detector = HandDetector(model_path="models/hand_landmarker.task", num_hands=2)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # mirror effect

        result = detector.detect(frame)

        if result:
            print(f"Detected {result.num_hands} hands")
            print("Handedness:", result.get_handedness_labels())

            frame = detector.draw_landmarks(frame, result, with_skeleton=True)

        cv2.imshow("Hand Detection Test - ESC to quit", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()