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
    
    # @property
    # def camera_dir(self) -> Path:
    #     """Camera-specific recording directory."""
    #     return self.media_dir
    
    # @property
    # def state_file(self) -> Path:
    #     """Motion detection state file."""
    #     return self.output_dir.parent / f".motion_state_{self.camera}.json"
    
    # ========================================================================
    # Motion Detection Parameters
    # ========================================================================
    
    @property
    def sample_fps(self) -> float:
        """Frame sampling rate for motion detection."""
        return float(os.environ.get("BIRBCAM_SAMPLE_FPS", "2"))
    
    @property
    def resize_width(self) -> int:
        """Target width for frame analysis."""
        return int(os.environ.get("BIRBCAM_RESIZE_W", "180"))
    
    @property
    def diff_threshold(self) -> int:
        """Pixel difference threshold (0-255)."""
        return int(os.environ.get("BIRBCAM_DIFF_THRESH", "25"))
    
    @property
    def min_changed_pixels(self) -> int:
        """Minimum changed pixels to consider motion."""
        return int(os.environ.get("BIRBCAM_MIN_CHANGED_PIXELS", "200"))
    
    @property
    def min_motion_frames(self) -> int:
        """Minimum consecutive frames with motion to trigger detection."""
        return int(os.environ.get("BIRBCAM_MIN_MOTION_FRAMES", "1"))
    
    @property
    def merge_gap(self) -> float:
        """Maximum seconds to merge nearby motion windows."""
        return float(os.environ.get("BIRBCAM_MERGE_GAP", "1.5"))
    
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
    
    @property
    def tail_near_end(self) -> float:
        """If motion ends within N seconds of video end, consider for joining."""
        return float(os.environ.get("BIRBCAM_TAIL_NEAR_END", "1.0"))
    
    @property
    def head_near_start(self) -> float:
        """If motion starts within N seconds of video start, consider for joining."""
        return float(os.environ.get("BIRBCAM_HEAD_NEAR_START", "1.0"))
    
    # ========================================================================
    # Performance Optimization
    # ========================================================================
    
    @property
    def use_ffmpeg_preprocessing(self) -> bool:
        """Use FFmpeg preprocessing for faster motion detection."""
        return os.environ.get("BIRBCAM_USE_FFMPEG_PREPROCESS", "1") == "1"
    
    @property
    def max_age_minutes(self) -> Optional[int]:
        """Only process files newer than N minutes (None = no limit)."""
        value = os.environ.get("BIRBCAM_MAX_AGE_MIN", "360")
        return int(value) if value and value != "0" else None
    
    @property
    def max_files_per_run(self) -> int:
        """Maximum number of files to process per run."""
        return int(os.environ.get("BIRBCAM_MAX_FILES", "20"))
    
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
    
    # ========================================================================
    # Debug and Testing
    # ========================================================================
    
    @property
    def dry_run(self) -> bool:
        """Dry run mode - analyze but don't create output files."""
        return os.environ.get("BIRBCAM_DRY_RUN", "0") == "1"
    
    # ========================================================================
    # Utility Methods
    # ========================================================================
    
    # def validate_directories(self, create: bool = True) -> bool:
    #     """Validate that all required directories exist or can be created."""
    #     dirs = [self.media_dir, self.output_dir, self.temp_dir]
        
    #     for directory in dirs:
    #         if not directory.exists():
    #             if create:
    #                 try:
    #                     directory.mkdir(parents=True, exist_ok=True)
    #                 except Exception as e:
    #                     print(f"Cannot create directory {directory}: {e}")
    #                     return False
    #             else:
    #                 print(f"Directory does not exist: {directory}")
    #                 return False
    #     return True
    
    def get_summary(self) -> dict:
        """Get a summary of current configuration values."""
        return {
            "camera": self.cameras,
            "log_level": self.log_level,
            "media_dir": str(self.media_dir),
            "output_dir": str(self.output_dir),
            "temp_dir": str(self.temp_dir),
            "sample_fps": self.sample_fps,
            "resize_width": self.resize_width,
            "use_ffmpeg_preprocessing": self.use_ffmpeg_preprocessing,
            "max_files_per_run": self.max_files_per_run,
            "dry_run": self.dry_run,
        }

    def get_summary_text(self) -> str:
        """Get a human-readable summary of current configuration."""
        summary = self.get_summary()
        lines = ["BirbCam Configuration Summary:"]
        for key, value in summary.items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)


# Global configuration instance
config = BirbCamConfig()


# Convenience functions for backward compatibility
def get_config(camera: Optional[str] = None) -> BirbCamConfig:
    """Get configuration instance, optionally for a specific camera."""
    if camera and camera != config.cameras:
        return BirbCamConfig(camera)
    return config
