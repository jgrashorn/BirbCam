"""
Camera settings management module.
Dynamically handles different camera types based on their actual config.
"""

import json
import os
import socket
from pathlib import Path

# Get the directory of this file
WEBAPP_DIR = os.path.dirname(os.path.abspath(__file__))

def load_cameras_config():
    """Load cameras.json configuration."""
    with open(os.path.join(WEBAPP_DIR, 'cameras.json'), 'r') as f:
        return json.load(f)

def load_settings_metadata():
    """Load camera_settings_metadata.json for UI hints (optional)."""
    try:
        with open(os.path.join(WEBAPP_DIR, 'camera_settings_metadata.json'), 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def get_camera_settings_endpoint(camera_name):
    """Get the settings endpoint (IP and port) for a camera."""
    cameras = load_cameras_config()
    return cameras.get(camera_name, {}).get('settings_endpoint', {})

def fetch_camera_config(camera_name):
    """
    Fetch the current configuration from a camera via TCP.
    Returns (success, config_dict or error_message).
    """
    endpoint = get_camera_settings_endpoint(camera_name)
    if not endpoint:
        return False, f"No endpoint configured for camera '{camera_name}'"
    
    ip = endpoint.get('ip')
    port = endpoint.get('port', 5005)
    
    try:
        # Connect to camera's settings server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((ip, port))
        
        # Send GET_CONFIG command
        sock.send(b"GET_CONFIG\n")
        
        # Receive response
        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
            if b'\n' in chunk:
                break
        
        sock.close()
        
        # Parse response
        response_str = response.decode('utf-8').strip()
        response_data = json.loads(response_str)
        
        if response_data.get('status') == 'ok':
            return True, response_data.get('config', {})
        else:
            return False, response_data.get('message', 'Unknown error')
    
    except socket.timeout:
        return False, f"Timeout connecting to camera at {ip}:{port}"
    except ConnectionRefusedError:
        return False, f"Connection refused by camera at {ip}:{port}"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON response: {e}"
    except Exception as e:
        return False, f"Error fetching config: {e}"

def send_camera_config(camera_name, config_dict):
    """
    Send new configuration to a camera via TCP.
    Returns (success, message).
    """
    endpoint = get_camera_settings_endpoint(camera_name)
    if not endpoint:
        return False, f"No endpoint configured for camera '{camera_name}'"
    
    ip = endpoint.get('ip')
    port = endpoint.get('port', 5005)
    
    try:
        # Connect to camera's settings server
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((ip, port))
        
        # Send SET_CONFIG command with JSON payload
        config_json = json.dumps(config_dict)
        command = f"SET_CONFIG:{config_json}\n"
        sock.send(command.encode('utf-8'))
        
        # Receive response
        response = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            response += chunk
            if b'\n' in chunk:
                break
        
        sock.close()
        
        # Parse response
        response_str = response.decode('utf-8').strip()
        response_data = json.loads(response_str)
        
        if response_data.get('status') == 'ok':
            return True, response_data.get('message', 'Configuration updated')
        else:
            return False, response_data.get('message', 'Unknown error')
    
    except socket.timeout:
        return False, f"Timeout connecting to camera at {ip}:{port}"
    except ConnectionRefusedError:
        return False, f"Connection refused by camera at {ip}:{port}"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON response: {e}"
    except Exception as e:
        return False, f"Error sending config: {e}"

def infer_camera_type_from_config(config):
    """
    Infer camera type based on the settings in the config.
    Returns 'picamera2', 'ffmpeg', or 'unknown'.
    """
    # Picamera2-specific settings
    picamera2_indicators = ['colorOffset_red', 'colorOffset_blue', 'bwSwitchingSensitivity', 'clrSwitchingSensitivity']
    # FFmpeg-specific settings  
    ffmpeg_indicators = ['inputFormat', 'preset', 'bitrate', 'enableAudio', 'audioDevice']
    
    picamera2_score = sum(1 for key in picamera2_indicators if key in config)
    ffmpeg_score = sum(1 for key in ffmpeg_indicators if key in config)
    
    if picamera2_score > ffmpeg_score:
        return 'picamera2'
    elif ffmpeg_score > picamera2_score:
        return 'ffmpeg'
    else:
        return 'unknown'

def infer_setting_type(value, setting_name=None, camera_type=None):
    """
    Infer the input type for a setting based on its value and context.
    Returns a dict with type and additional metadata.
    """
    if isinstance(value, bool):
        return {'type': 'boolean'}
    elif isinstance(value, int):
        return {'type': 'number', 'step': 1}
    elif isinstance(value, float):
        return {'type': 'number', 'step': 0.1}
    elif isinstance(value, str):
        # Check if it's a device path
        if value.startswith('/dev/') or value.startswith('hw:'):
            return {'type': 'text', 'pattern': '.*'}
        # Check if it's a bitrate
        elif value.endswith('k') or value.endswith('M'):
            return {'type': 'text', 'pattern': '[0-9]+[kM]?'}
        else:
            return {'type': 'text'}
    else:
        return {'type': 'text'}

def apply_preset_constraints(setting_name, metadata, camera_type):
    """
    Apply preset value constraints for specific settings based on camera type.
    Converts certain settings to select dropdowns with fixed options.
    """
    # Resolution presets based on camera type
    if setting_name in ['width', 'height']:
        if camera_type == 'picamera2':
            # Picamera2 supports specific resolutions
            if setting_name == 'width':
                metadata['type'] = 'select'
                metadata['options'] = [1296, 1920]
                metadata['linked_setting'] = 'height'
            elif setting_name == 'height':
                metadata['type'] = 'select'
                metadata['options'] = [972, 1080]
                metadata['linked_setting'] = 'width'
        elif camera_type == 'ffmpeg':
            # USB camera best at 720p
            if setting_name == 'width':
                metadata['type'] = 'select'
                metadata['options'] = [640, 1280, 1920]
                metadata['linked_setting'] = 'height'
            elif setting_name == 'height':
                metadata['type'] = 'select'
                metadata['options'] = [480, 720, 1080]
                metadata['linked_setting'] = 'width'
    
    # Framerate presets for FFmpeg
    elif setting_name == 'framerate' and camera_type == 'ffmpeg':
        if metadata.get('type') == 'number':
            metadata['type'] = 'select'
            metadata['options'] = [15, 25, 30]
    
    return metadata

def get_settings_metadata_for_camera(camera_name):
    """
    Get settings metadata for a camera by fetching its current config
    and optionally enhancing with predefined metadata.
    Returns dict with camera type, current settings, and UI metadata.
    """
    # Fetch current config from camera
    success, config = fetch_camera_config(camera_name)
    
    if not success:
        return {
            'error': config,
            'settings': {}
        }
    
    # Infer camera type from config
    camera_type = infer_camera_type_from_config(config)
    
    # Try to get configured camera type from cameras.json as fallback
    cameras = load_cameras_config()
    if camera_type == 'unknown' and camera_name in cameras:
        camera_type = cameras[camera_name].get('camera_type', 'unknown')
    
    # Load optional metadata for UI enhancements
    all_metadata = load_settings_metadata()
    
    # Build settings metadata
    settings_metadata = {}
    
    for setting_name, setting_value in config.items():
        # Start with inferred metadata
        metadata = infer_setting_type(setting_value, setting_name, camera_type)
        metadata['current_value'] = setting_value
        
        # Enhance with predefined metadata if available
        if camera_type in all_metadata:
            type_meta = all_metadata[camera_type].get('settings', {})
            if setting_name in type_meta:
                predefined = type_meta[setting_name]
                # Merge predefined metadata (label, description, min, max, etc.)
                metadata.update({k: v for k, v in predefined.items() if k not in ['current_value']})
        
        # Apply preset constraints
        metadata = apply_preset_constraints(setting_name, metadata, camera_type)
        
        # Add a default label if not provided
        if 'label' not in metadata:
            # Convert camelCase or snake_case to Title Case
            label = setting_name.replace('_', ' ').replace('  ', ' ')
            label = ''.join([' ' + c if c.isupper() else c for c in label]).strip()
            metadata['label'] = label.title()
        
        settings_metadata[setting_name] = metadata
    
    return {
        'camera_type': camera_type,
        'settings': settings_metadata
    }

def get_all_cameras_info():
    """
    Get comprehensive information about all cameras by fetching their configs.
    Returns dict with camera names as keys and their metadata as values.
    """
    cameras = load_cameras_config()
    result = {}
    
    for camera_name in cameras:
        result[camera_name] = {
            'endpoint': get_camera_settings_endpoint(camera_name),
            'metadata': get_settings_metadata_for_camera(camera_name)
        }
    
    return result

def validate_setting(camera_name, setting_name, value):
    """
    Validate a setting value for a specific camera.
    Returns (is_valid, error_message).
    """
    metadata = get_settings_metadata_for_camera(camera_name)
    
    if 'error' in metadata:
        return False, f"Cannot validate: {metadata['error']}"
    
    if setting_name not in metadata['settings']:
        # If setting doesn't exist in current config, allow it (new setting)
        return True, None
    
    setting_def = metadata['settings'][setting_name]
    setting_type = setting_def['type']
    
    # Type-specific validation
    if setting_type == 'number':
        try:
            num_value = float(value)
            if 'min' in setting_def and num_value < setting_def['min']:
                return False, f"Value must be >= {setting_def['min']}"
            if 'max' in setting_def and num_value > setting_def['max']:
                return False, f"Value must be <= {setting_def['max']}"
            return True, None
        except (ValueError, TypeError):
            return False, "Value must be a number"
    
    elif setting_type == 'boolean':
        if not isinstance(value, bool):
            return False, "Value must be true or false"
        return True, None
    
    elif setting_type == 'select':
        if 'options' in setting_def:
            # Convert to same type for comparison
            options = setting_def['options']
            if value not in options and str(value) not in [str(o) for o in options]:
                return False, f"Value must be one of: {', '.join(map(str, options))}"
        return True, None
    
    elif setting_type == 'text':
        if not isinstance(value, str):
            return False, "Value must be a string"
        return True, None
    
    return True, None
