#!/usr/bin/env python3
"""
BirbCam Configuration Setup Helper

This script helps set up the configuration for BirbCam and validates that everything works correctly.
"""

import os
import sys
from pathlib import Path

def check_env_file():
    """Check if .env file exists and is properly formatted."""
    env_file = Path(__file__).parent / ".env"
    
    if not env_file.exists():
        print("❌ .env file not found!")
        print(f"   Expected location: {env_file}")
        print("   Run: cp .env.example .env")
        return False
    
    print(f"✅ Found .env file: {env_file}")
    
    # Check if required variables are set
    required_vars = ['BIRBCAM_CAMERAS_JSON', 'BIRBCAM_MEDIA_DIR', 'BIRBCAM_OUTPUT_DIR']
    missing_vars = []
    
    with open(env_file, 'r') as f:
        content = f.read()
        for var in required_vars:
            if f"{var}=" not in content or f"{var}=#" in content:
                missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing or commented required variables: {missing_vars}")
        print("   Please edit .env file and set proper values")
        return False
    else:
        print("✅ All required variables found in .env file")
        return True

def load_and_test_config():
    """Load configuration and test that it works."""
    # Load .env file
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
        print("✅ Environment variables loaded")
    
    try:
        from config import config
        print("✅ Configuration module imported successfully")
        
        # Test key configuration values
        print(f"   Cameras: {config.cameras}")
        print(f"   Media directory: {config.media_dir}")
        print(f"   Output directory: {config.output_dir}")
        print(f"   Log level: {config.log_level}")
        
        return True
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def check_directories():
    """Check if configured directories exist and are accessible."""
    try:
        from config import config
        
        print("\n=== Directory Check ===")
        
        # Check media directory (input)
        if config.media_dir.exists():
            print(f"✅ Media directory exists: {config.media_dir}")
        else:
            print(f"⚠️  Media directory missing: {config.media_dir}")
            print("   This directory should contain MP4 files from MediaMTX")
        
        # Check output directory (will be created if needed)
        if config.output_dir.exists():
            print(f"✅ Output directory exists: {config.output_dir}")
        else:
            print(f"📁 Output directory will be created: {config.output_dir}")
        
        # Check temp directory
        if config.temp_dir.exists():
            print(f"✅ Temp directory exists: {config.temp_dir}")
        else:
            print(f"⚠️  Temp directory missing: {config.temp_dir}")
            print("   This should be on fast storage for optimal performance")
        
        return True
    except Exception as e:
        print(f"❌ Directory check failed: {e}")
        return False

def show_usage():
    """Show usage instructions."""
    print("\n=== Usage Instructions ===")
    print("1. Make sure your .env file is properly configured")
    print("2. Run scripts with:")
    print("   cd /home/birb/BirbCam/Webapp")
    print("   python3 motiondetection.py    # for motion detection")
    print("   python3 webapp.py            # for web interface")
    print("\n3. Or run with explicit environment loading:")
    print("   bash -c 'set -a && source .env && set +a && python3 motiondetection.py'")
    print("\n4. For systemd services, use EnvironmentFile:")
    print("   EnvironmentFile=/home/birb/BirbCam/Webapp/.env")

def main():
    """Main setup and validation routine."""
    print("🏗️  BirbCam Configuration Setup & Validation")
    print("=" * 50)
    
    steps = [
        ("Environment File", check_env_file),
        ("Configuration Loading", load_and_test_config),
        ("Directory Check", check_directories),
    ]
    
    results = []
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}")
        print("-" * 30)
        result = step_func()
        results.append(result)
        
        if not result:
            print(f"❌ {step_name} failed - please fix issues before continuing")
            break
    
    print(f"\n🏁 Setup Results: {sum(results)}/{len(results)} steps completed")
    
    if all(results):
        print("🎉 Configuration is ready! BirbCam should work correctly.")
        show_usage()
        return 0
    else:
        print("🔧 Please fix the issues above and run this script again.")
        return 1

if __name__ == "__main__":
    sys.exit(main())