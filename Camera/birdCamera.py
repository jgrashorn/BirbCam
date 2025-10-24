import json
import logging
import socket
import threading
import time

from picamera2.outputs import FileOutput

logger = logging.getLogger(__name__)

# Global config state
_config = None
_config_lock = threading.Lock()
_config_changed = threading.Event()

def readConfig():
    global _config
    try:
        with open("config.txt") as f:
            configData = f.read()
        
        parsedConfig = json.loads(configData)
        
        with _config_lock:
            _config = parsedConfig
        
        return parsedConfig
    except Exception as e:
        logger.error(f"Failed to read config: {e}")
        return {}

def writeConfig(config):
    """Write configuration to file and signal change."""
    global _config
    try:
        with open("config.txt", "w") as f:
            json.dump(config, f, indent=4)
        
        with _config_lock:
            _config = config.copy()
        
        _config_changed.set()
        logger.info("Configuration updated and saved")
        return True
    except Exception as e:
        logger.error(f"Failed to write config: {e}")
        return False

def getCurrentConfig():
    """Get current configuration safely."""
    with _config_lock:
        return _config.copy() if _config else {}

def waitForConfigChange(timeout=None):
    """Wait for configuration change event."""
    result = _config_changed.wait(timeout)
    if result:
        _config_changed.clear()
    return result

def handleSettingsClient(client_socket, address):
    """Handle TCP client for settings changes."""
    logger.info(f"Settings client connected from {address}")
    
    try:
        while True:
            # Receive data
            data = client_socket.recv(1024).decode('utf-8').strip()
            if not data:
                break
            
            logger.info(f"Received command: {data}")
            
            if data == "GET_CONFIG":
                # Send current configuration
                config = getCurrentConfig()
                response = json.dumps({"status": "ok", "config": config})
                client_socket.send((response + "\n").encode('utf-8'))
                
            elif data.startswith("SET_CONFIG:"):
                # Set new configuration
                try:
                    config_json = data[11:]  # Remove "SET_CONFIG:" prefix
                    new_config = json.loads(config_json)
                    
                    # Validate configuration keys (basic validation)
                    required_keys = ["serverIP", "port", "name", "rtspPort", "width", "height"]
                    if all(key in new_config for key in required_keys):
                        if writeConfig(new_config):
                            response = json.dumps({"status": "ok", "message": "Configuration updated"})
                        else:
                            response = json.dumps({"status": "error", "message": "Failed to save configuration"})
                    else:
                        response = json.dumps({"status": "error", "message": "Missing required configuration keys"})
                        
                except json.JSONDecodeError as e:
                    response = json.dumps({"status": "error", "message": f"Invalid JSON: {str(e)}"})
                except Exception as e:
                    response = json.dumps({"status": "error", "message": f"Error: {str(e)}"})
                
                client_socket.send((response + "\n").encode('utf-8'))
                
            elif data == "PING":
                response = json.dumps({"status": "ok", "message": "pong"})
                client_socket.send((response + "\n").encode('utf-8'))
                
            else:
                response = json.dumps({"status": "error", "message": "Unknown command"})
                client_socket.send((response + "\n").encode('utf-8'))
                
    except Exception as e:
        logger.error(f"Error handling settings client {address}: {e}")
    finally:
        client_socket.close()
        logger.info(f"Settings client {address} disconnected")

def startSettingsServer(port=5005):
    """Start TCP server for settings management."""
    def server_thread():
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            server_socket.bind(('0.0.0.0', port))
            server_socket.listen(5)
            logger.info(f"Settings server listening on port {port}")
            
            while True:
                try:
                    client_socket, address = server_socket.accept()
                    # Handle each client in a separate thread
                    client_thread = threading.Thread(
                        target=handleSettingsClient,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except Exception as e:
                    logger.error(f"Error accepting client: {e}")
                    time.sleep(1)
                    
        except Exception as e:
            logger.error(f"Settings server error: {e}")
        finally:
            server_socket.close()
    
    # Start server in daemon thread
    thread = threading.Thread(target=server_thread, daemon=True)
    thread.start()
    return thread

