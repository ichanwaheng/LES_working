"""Continuous motion of a form-found membrane submerged in fluid flow.

After form-finding, this package advances a lumped-mass membrane in the
LES/PISO flow and writes a simulation video as the primary output.
"""

from .capture import ContinuousMotionCapture, ContinuousMotionHistory
from .video import write_simulation_video

__all__ = [
    "ContinuousMotionCapture",
    "ContinuousMotionHistory",
    "write_simulation_video",
]
