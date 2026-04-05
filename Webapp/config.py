"""
BirbCam Configuration Management

Centralized configuration for all BirbCam components.
This module handles environment variables and provides type-safe configuration access.
"""

import os
from pathlib import Path
from typing import Optional, Union
import json

class BirbCamConfig:
    """Centralized configuration for BirbCam system."""
    
    def __init__(self, camera: Optional[str] = None):
        """Initialize configuration, optionally overriding camera name."""
        self._cameras_json = os.environ.get("BIRBCAM_CAMERAS_JSON", "cameras.json")
        with open(self._cameras_json, 'r') as f:
            self._cameras = json.load(f)

        if camera:
            if camera not in self._cameras:
                raise ValueError(f"Camera '{camera}' not found in cameras.json")
            self._cameras = self._cameras[camera]
    # ========================================================================
    # Core System Settings
    # ========================================================================
    
    @property
    def cameras(self) -> str:
        """Camera identifier (should match MediaMTX stream path)."""
        return self._cameras
    
    @property
    def log_level(self) -> str:
        """Logging level for motion detection script."""
        return os.environ.get("BIRBCAM_LOG", "DEBUG").upper()

    # ============================================================================
    # Stream and Server Settings
    # ============================================================================
    
    @property
    def hlspath(self) -> str:
        """HLS stream base path."""
        return os.environ.get("BIRBCAM_HLS_PORT", "8888")
    
    @property
    def hlsip(self) -> str:
        """HLS stream server IP address."""
        return os.environ.get("BIRBCAM_HLS_IP", "localhost")
    # ========================================================================
    # Directory Paths
    # ========================================================================
    
    @property
    def media_dir(self) -> Path:
        """Directory where MediaMTX writes recorded MP4 files."""
        default = "/home/birb/mediamtx/recordings"
        return Path(os.environ.get("BIRBCAM_MEDIA_DIR", default))
    
    @property
    def output_dir(self) -> Path:
        """Directory where processed motion clips are stored."""
        default = f"/home/birb/BirbCam/Webapp/recordings"
        return Path(os.environ.get("BIRBCAM_OUTPUT_DIR", default))
    
    @property
    def temp_dir(self) -> Path:
        """Temporary directory for FFmpeg preprocessing."""
        default = "/tmp/birbcam"
        return Path(os.environ.get("BIRBCAM_TEMP_DIR", default))
    
    # ========================================================================
    # Motion Detection Parameters
    # ========================================================================
    

    @property
    def merge_gap(self) -> float:
        """Maximum seconds to merge nearby motion windows."""
        return float(os.environ.get("BIRBCAM_MERGE_GAP", "3.0"))
    
    @property
    def min_event_duration(self) -> float:
        """Minimum motion event duration (seconds) — shorter events are discarded."""
        return float(os.environ.get("BIRBCAM_MIN_EVENT_DURATION", "0.5"))
    
    @property
    def max_event_duration(self) -> float:
        """Maximum duration (seconds) of a single motion event clip."""
        return float(os.environ.get("BIRBCAM_MAX_EVENT_DURATION", "120"))
    
    @property
    def event_retention_days(self) -> int:
        """Number of days to retain motion event records."""
        return int(os.environ.get("BIRBCAM_EVENT_RETENTION_DAYS", "7"))
    
    # ========================================================================
    # Clip Export Behavior
    # ========================================================================
    
    @property
    def pre_buffer(self) -> float:
        """Seconds to include before motion starts."""
        return float(os.environ.get("BIRBCAM_PRE_BUFFER", "2.0"))
    
    @property
    def post_buffer(self) -> float:
        """Seconds to include after motion ends."""
        return float(os.environ.get("BIRBCAM_POST_BUFFER", "3.0"))
    
    # ========================================================================
    # Performance Optimization
    # ========================================================================
    
    @property
    def max_age_minutes(self) -> Optional[int]:
        """Only process files newer than N minutes (None = no limit)."""
        value = os.environ.get("BIRBCAM_MAX_AGE_MIN", "360")
        return int(value) if value and value != "0" else None
    
    # ========================================================================
    # File Stability Detection
    # ========================================================================
    
    @property
    def stable_min_age(self) -> float:
        """Minimum age (seconds) before a file is considered stable."""
        return float(os.environ.get("BIRBCAM_STABLE_MIN_AGE", "5.0"))
    
    @property
    def stable_poll_seconds(self) -> float:
        """Time to wait between size checks when determining file stability."""
        return float(os.environ.get("BIRBCAM_STABLE_POLL_SECONDS", "1.0"))

# Global configuration instance
config = BirbCamConfig()


# Convenience functions for backward compatibility
def get_config(camera: Optional[str] = None) -> BirbCamConfig:
    """Get configuration instance, optionally for a specific camera."""
    if camera and camera != config.cameras:
        return BirbCamConfig(camera)
    return config
