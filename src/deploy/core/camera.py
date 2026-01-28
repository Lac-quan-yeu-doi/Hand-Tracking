import cv2
import numpy as np
from typing import Tuple, Optional

from .components import Region

class CameraManager:
    def __init__(self, flip_horizontal: bool = True):
        """
        Args:
            flip_horizontal: Whether to mirror the image (typical for webcams)
        """
        self.cap: Optional[cv2.VideoCapture] = None
        self.flip_horizontal = flip_horizontal
        self.roi: Optional[Region] = None
        self.frame_width: int = 0
        self.frame_height: int = 0

    def start_capture(
        self,
        camera_id: int = 0,
        resolution: Optional[Tuple[int, int]] = None
    ) -> bool:
        """
        Args:
            camera_id: Camera index (0 = default webcam, 1 = external, etc.)
            resolution: Desired (width, height). May not be exactly matched by hardware.
            
        Returns:
            bool: True if camera opened successfully
        """
        
        # Release any existing capture
        if self.cap is not None:
            self.stop_capture()

        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            print(f"Error: Could not open camera with index {camera_id}")
            self.cap = None
            return False

        # Try to set requested resolution (not all cameras support it)
        if resolution is not None:
            width, height = resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Get actual resolution after setting
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.frame_width <= 0 or self.frame_height <= 0:
            print(f"Error: Cannot initialize frame size - Width error: {self.frame_width <= 0}, Height error: {self.frame_height <= 0}")
            self.stop_capture()
            return False

        print(f"Camera started: {self.frame_width}x{self.frame_height}")
        return True

    def get_frame(self, visualize=None) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
        """        
        Get and preprocess frame.
        - Horizontal flip (mirror) if enabled
        - Set ROI
        
        Returns:
            Tuple[BGR frame (np.ndarray), ROI mask (np.ndarray)] or None if frame not available
        """
        if self.cap is None or not self.cap.isOpened():
            return None

        success, frame = self.cap.read()
        if not success:
            return None

        # Mirror effect (most common for webcams)
        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)

        roi_frame = None
        roi_pixels = self._roi_get(frame)

        if roi_pixels is not None:
            x1, y1, x2, y2 = roi_pixels

            # Extract ROI for processing
            roi_frame = frame[y1:y2, x1:x2].copy()

            # Draw ROI on the full frame
            if visualize is not None:
                self._roi_draw(frame, roi_pixels)

        return frame, roi_frame        

    def _roi_get(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Convert normalized ROI to absolute pixel coordinates
        Args:
            frame (np.ndarray): Input frame

        Returns:
            Optional[Tuple[int, int, int, int]]: (x1, y1, x2, y2)
        """
        if self.roi is None:
            return None
        h, w = frame.shape[:2]

        x1 = int(self.roi.x * w)
        y1 = int(self.roi.y * h)
        x2 = int((self.roi.x + self.roi.width) * w)
        y2 = int((self.roi.y + self.roi.height) * h)

        # Clamp
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        return x1, y1, x2, y2
    
    def _roi_draw(self, frame: np.ndarray, roi_pixels) -> None:
        """ROI rectangle on frame"""
        x1, y1, x2, y2 = roi_pixels
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color=(0, 255, 0),   # green box
            thickness=2
        )    

    def _roi_crop(self, frame: np.ndarray) -> np.ndarray:
        """Internal method: crop frame to current ROI"""
        if self.roi is None:
            return frame

        h, w = frame.shape[:2]

        x_start = int(self.roi.x * w)
        y_start = int(self.roi.y * h)
        x_end = int((self.roi.x + self.roi.width) * w)
        y_end = int((self.roi.y + self.roi.height) * h)

        # Clamp to frame boundaries
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        x_end = min(w, x_end)
        y_end = min(h, y_end)

        # Prevent empty/invalid crop
        if x_end <= x_start or y_end <= y_start:
            print("Warning: Invalid ROI dimensions → returning full frame")
            return frame

        return frame[y_start:y_end, x_start:x_end]

    def set_region_of_interest(self, region: Region) -> None:
        if not (0 <= region.x <= 1 and 0 <= region.y <= 1 and
                0 < region.width <= 1 and 0 < region.height <= 1):
            print("Warning: Invalid ROI values (must be in [0,1]). Ignoring.")
            return

        self.roi = region
        print(f"Region of interest updated: {region}")

    def get_current_resolution(self) -> Tuple[int, int]:
        return (self.frame_width, self.frame_height)

    def stop_capture(self) -> None:
        """Release the camera resource"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            print("Camera released")

    def toggle_flip(self, enabled: bool) -> None:
        self.flip_horizontal = enabled

    def __del__(self):
        """Auto-release camera when object is destroyed"""
        self.stop_capture()


# ──────────────────────────────────────────────────────────────
# Quick test / example usage
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cam = CameraManager(flip_horizontal=True)
    
    if cam.start_capture(camera_id=0, resolution=(1280, 720)):
        cam.set_region_of_interest(Region(x=0.15, y=0.15, width=0.7, height=0.7))
        cam.toggle_flip(True)
        while True:
            frame, roi = cam.get_frame(visualize=True)
            if frame is None:
                break

            cv2.imshow("CameraManager - Press ESC to quit", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cam.stop_capture()
        cv2.destroyAllWindows()