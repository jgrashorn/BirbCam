# Dynamic Camera Settings System

## Overview
The webapp now dynamically fetches and displays settings from each camera based on their actual configuration files.

## How It Works

### 1. Camera Side
Each camera (picamera2 or FFmpeg) has:
- `config.txt` - Camera-specific settings
- `birdCamera.py` - TCP server that responds to `GET_CONFIG` and `SET_CONFIG` commands

### 2. Webapp Side
- **camera_settings.py** - Fetches config from cameras and infers UI metadata
- **webapp.py** - Routes use camera_settings to get/set configuration
- **settings.html** - Dynamically renders form based on actual settings

### 3. Configuration Flow

**Getting Settings:**
1. User visits `/settings/<camera_name>`
2. Webapp calls `camera_settings.get_settings_metadata_for_camera()`
3. This sends `GET_CONFIG` TCP command to camera
4. Camera responds with its current `config.txt`
5. System infers input types (number, boolean, text, select)
6. Template renders appropriate HTML controls

**Setting Configuration:**
1. User submits form
2. Webapp validates values against metadata
3. Sends `SET_CONFIG` command to camera with JSON
4. Camera updates `config.txt` and applies changes

## Benefits

✅ **No manual synchronization** - Settings list comes from camera
✅ **Type-safe** - Automatic inference of input types
✅ **Future-proof** - Add new settings to camera config, they appear automatically
✅ **Camera-specific** - Each camera type shows only its relevant settings
✅ **Enhanced UX** - Optional metadata.json provides nice labels and descriptions

## Adding New Settings

Simply add to the camera's `config.txt`:
```json
{
    "existingSetting": 123,
    "newSetting": "value"
}
```

The webapp will automatically:
- Display it in the settings page
- Infer appropriate input type
- Validate changes
- Send updates back to camera

## Metadata Enhancement

`camera_settings_metadata.json` provides optional UI enhancements:
- Nice labels and descriptions
- Min/max ranges for numbers
- Dropdown options for selects
- Help text

But it's **optional** - settings work without it!
