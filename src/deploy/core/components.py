from typing import NamedTuple

class Region(NamedTuple):
    """Normalized region of interest (all values in range 0.0 to 1.0)"""
    x: float      # left position (0.0 = left edge)
    y: float      # top position
    width: float  # width fraction of full frame
    height: float # height fraction of full frame