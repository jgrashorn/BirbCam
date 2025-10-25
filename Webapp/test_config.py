"""
Quick test to validate the configuration migration works correctly.
"""

import os
import sys
from pathlib import Path

def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print(f"✓ Loaded environment from {env_file}")
    else:
        print(f"⚠ No .env file found at {env_file}")

# Load environment variables first
load_env_file()

from config import config

# Use centralized configuration
CAMERA = config.camera
MEDIAMTX_RECORD_DIR = config.media_dir
CAMERA_DIR = (MEDIAMTX_RECORD_DIR / CAMERA)
OUTPUT_DIR = config.output_dir
STATE_FILE = OUTPUT_DIR.parent / f".motion_state_{CAMERA}.json"

def main():
    print("=== BirbCam Configuration Test ===")
    
    # Show environment status
    print("\n--- Environment Variables ---")
    important_vars = ['BIRBCAM_MEDIA_DIR', 'BIRBCAM_OUTPUT_DIR', 'BIRBCAM_CAMERA']
    for var in important_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"{var}: {value}")
    
    print("\n--- Configuration Values ---")
    print(f"Camera: {CAMERA}")
    print(f"MediaMTX Record Dir: {MEDIAMTX_RECORD_DIR}")
    print(f"Camera Dir: {CAMERA_DIR}")
    print(f"Output Dir: {OUTPUT_DIR}")
    print(f"State File: {STATE_FILE}")
    print(f"Log Level: {config.log_level}")
    
    print("\n--- Validation ---")
    # Check if config matches environment
    env_media_dir = os.environ.get('BIRBCAM_MEDIA_DIR', '')
    env_camera = os.environ.get('BIRBCAM_CAMERA', '')
    
    if str(MEDIAMTX_RECORD_DIR) == env_media_dir:
        print("✓ Media directory matches environment variable")
    else:
        print(f"✗ Media directory mismatch: config={MEDIAMTX_RECORD_DIR}, env={env_media_dir}")
    
    if CAMERA == env_camera:
        print("✓ Camera matches environment variable") 
    else:
        print(f"✗ Camera mismatch: config={CAMERA}, env={env_camera}")
    
    print("\nConfiguration test completed successfully.")

if __name__ == "__main__":
    main()