#!/usr/bin/env python3
"""Test script to validate that both webapp.py and motiondetection.py can import config correctly."""

import os
import sys
from pathlib import Path

# Set up path
webapp_dir = Path(__file__).parent
sys.path.insert(0, str(webapp_dir))

def test_motiondetection_config():
    """Test motiondetection.py config loading."""
    print("=== Testing motiondetection.py ===")
    
    # Load .env file manually for this test
    env_file = webapp_dir / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    try:
        from config import config
        
        # Test the same variables motiondetection.py uses
        CAMERA = config.camera
        MEDIAMTX_RECORD_DIR = config.media_dir
        SAMPLE_FPS = config.sample_fps
        
        print(f"✓ Camera: {CAMERA}")
        print(f"✓ Media directory: {MEDIAMTX_RECORD_DIR}")
        print(f"✓ Sample FPS: {SAMPLE_FPS}")
        
        # Verify against environment
        expected_media_dir = os.environ.get('BIRBCAM_MEDIA_DIR', '')
        if str(MEDIAMTX_RECORD_DIR) == expected_media_dir:
            print(f"✓ Media directory matches environment: {expected_media_dir}")
            return True
        else:
            print(f"✗ Media directory mismatch: got {MEDIAMTX_RECORD_DIR}, expected {expected_media_dir}")
            return False
            
    except Exception as e:
        print(f"✗ Error loading motiondetection config: {e}")
        return False

def test_webapp_config():
    """Test webapp.py config loading."""
    print("\n=== Testing webapp.py ===")
    
    try:
        from config import config
        
        # Test the same variables webapp.py uses
        RECORD_ROOT = config.media_dir
        WEB_RECORD_ROOT = config.output_dir
        
        print(f"✓ Record root: {RECORD_ROOT}")
        print(f"✓ Web record root: {WEB_RECORD_ROOT}")
        
        # Verify against environment
        expected_media_dir = os.environ.get('BIRBCAM_MEDIA_DIR', '')
        if str(RECORD_ROOT) == expected_media_dir:
            print(f"✓ Record root matches environment: {expected_media_dir}")
            return True
        else:
            print(f"✗ Record root mismatch: got {RECORD_ROOT}, expected {expected_media_dir}")
            return False
            
    except Exception as e:
        print(f"✗ Error loading webapp config: {e}")
        return False

def main():
    print("=== BirbCam Configuration Integration Test ===")
    
    results = []
    results.append(test_motiondetection_config())
    results.append(test_webapp_config())
    
    print(f"\n=== Results: {sum(results)}/{len(results)} passed ===")
    
    if all(results):
        print("🎉 All configuration tests passed!")
        return 0
    else:
        print("❌ Some configuration tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())