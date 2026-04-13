from enum import Enum


# Termination reason tracking for episode analysis
class TerminationReason(Enum):
    SUCCESS = "success"  # Agent crossed the finish line
    OFF_TRACK = "off_track"  # Agent drove off the track
    TIMEOUT = "timeout"  # Episode ended due to maximum steps
    EARLY_TERMINATION = "early"  # Custom early termination triggered
    UNKNOWN = "unknown"  # Could not determine termination reason
