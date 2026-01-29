import numpy as np


class FingerTracker:
    """
    Tracks a single finger over time and computes velocity (gradient vector)
    """

    def __init__(self, min_speed=0.02):
        self.prev_pos = None   # (x, y) normalized
        self.prev_time = None  # ms
        self.min_speed = min_speed

    def update(self, x, y, timestamp_ms):
        """
        Update finger position and compute velocity vector.

        Returns:
            velocity (vx, vy) or None if not enough data
        """
        if self.prev_pos is None:
            self.prev_pos = (x, y)
            self.prev_time = timestamp_ms
            return None

        dx = x - self.prev_pos[0]
        dy = y - self.prev_pos[1]
        dt = (timestamp_ms - self.prev_time) / 1000.0  # seconds

        self.prev_pos = (x, y)
        self.prev_time = timestamp_ms

        if dt <= 0:
            return None

        vx = dx / dt
        vy = dy / dt

        speed = np.sqrt(vx * vx + vy * vy)
        if speed < self.min_speed:
            return None

        return vx, vy, speed


def classify_swipe(vx, vy):
    """
    Classify swipe direction from velocity vector.
    Screen coordinate system:
      +x → right
      +y → down
    """
    if abs(vx) > abs(vy):
        return "RIGHT" if vx > 0 else "LEFT"
    else:
        return "DOWN" if vy > 0 else "UP"
