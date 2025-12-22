import time
import os
import logging
import socket
import threading
import subprocess
import signal
import re

import birdCamera

logger = logging.getLogger(__name__)

def _rtsp_up(host, port, timeout=1.0):
    """Check if RTSP server is reachable."""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False

def detect_camera_capabilities(device):
    """Detect camera supported resolutions and framerates."""
    try:
        result = subprocess.run(
            ['v4l2-ctl', '--device', device, '--list-formats-ext'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            logger.info(f"Camera capabilities for {device}:")
            logger.info(result.stdout)
            
            # Parse for MJPEG resolutions
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if 'MJPEG' in line or 'Motion-JPEG' in line:
                    logger.info("MJPEG format found, listing resolutions:")
                    for j in range(i+1, min(i+20, len(lines))):
                        if 'Size:' in lines[j] or 'Interval:' in lines[j]:
                            logger.info(lines[j])
        else:
            logger.warning(f"Could not detect camera capabilities: {result.stderr}")
            
    except FileNotFoundError:
        logger.warning("v4l2-ctl not found, install v4l-utils to detect camera capabilities")
    except Exception as e:
        logger.error(f"Error detecting camera capabilities: {e}")

def runCamera():
    """Main camera streaming function using FFmpeg with USB camera."""
    
    os.environ['PULSE_SERVER'] = "/run/user/1000/pulse/native"

    logging.basicConfig(
        filename='birb.log',
        format='%(asctime)s %(levelname)s %(module)s %(funcName)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Load server configuration first
    server_config = birdCamera.readServerConfig()
    
    # Load camera configuration
    config = birdCamera.readConfig()
    
    # Start settings server
    settings_port = server_config.get("settingsPort", 5005)
    birdCamera.startSettingsServer(settings_port)

    # Video device - adjust if needed
    video_device = config.get("videoDevice", "/dev/video0")
    
    # Log camera capabilities on startup
    logger.info(f"Initializing camera on {video_device}")
    detect_camera_capabilities(video_device)

    # State management
    ffmpeg_process = None
    process_lock = threading.Lock()
    streaming = {"running": False}
    restart_stream = threading.Event()
    
    # Video device - adjust if needed
    video_device = config.get("videoDevice", "/dev/video0")
    
    def build_ffmpeg_command(cfg, srv_cfg):
        """Build FFmpeg command based on current configuration."""
        width = cfg.get("width", 1280)
        height = cfg.get("height", 960)
        framerate = cfg.get("framerate", 15)
        bitrate = cfg.get("bitrate", "2000k")
        preset = cfg.get("preset", "ultrafast")
        
        rtsp_url = f'rtsp://{srv_cfg["serverIP"]}:{srv_cfg["rtspPort"]}/{srv_cfg["name"]}'
        
        # FFmpeg command for USB camera (video only, no audio)
        cmd = [
            'ffmpeg',
            '-f', 'v4l2',
            '-input_format', 'mjpeg',
            '-video_size', f'{width}x{height}',
            '-framerate', str(framerate),
            '-i', video_device,
            # Video encoding options
            '-c:v', 'libx264',
            '-preset', preset,
            '-b:v', bitrate,
            '-pix_fmt', 'yuv420p',
            '-colorspace', 'bt709',
            '-color_range', 'tv',
            # Error recovery options
            '-err_detect', 'ignore_err',
            '-max_delay', '500000',
            # RTSP output
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',
            rtsp_url
        ]
        
        return cmd, rtsp_url
    
    def start_stream(cfg, srv_cfg):
        """Start FFmpeg streaming process."""
        nonlocal ffmpeg_process
        
        try:
            cmd, rtsp_url = build_ffmpeg_command(cfg, srv_cfg)
            logger.info(f"Starting FFmpeg stream to {rtsp_url}")
            logger.info(f"Command: {' '.join(cmd)}")
            
            # Start FFmpeg process
            ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                preexec_fn=os.setsid  # Create new process group for clean termination
            )
            
            streaming["running"] = True
            logger.info(f"FFmpeg process started with PID {ffmpeg_process.pid}")
            
        except Exception as e:
            logger.error(f"Failed to start FFmpeg stream: {e}", exc_info=True)
            ffmpeg_process = None
            streaming["running"] = False
    
    def stop_stream():
        """Stop FFmpeg streaming process gracefully."""
        nonlocal ffmpeg_process
        
        if ffmpeg_process is None:
            streaming["running"] = False
            return
        
        try:
            logger.info(f"Stopping FFmpeg process (PID {ffmpeg_process.pid})")
            
            # Send SIGTERM for graceful shutdown
            try:
                os.killpg(os.getpgid(ffmpeg_process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            
            # Wait up to 5 seconds for graceful shutdown
            try:
                ffmpeg_process.wait(timeout=5)
                logger.info("FFmpeg process terminated gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't stop
                logger.warning("FFmpeg didn't stop gracefully, forcing termination")
                try:
                    os.killpg(os.getpgid(ffmpeg_process.pid), signal.SIGKILL)
                    ffmpeg_process.wait(timeout=2)
                except Exception as e:
                    logger.error(f"Error force-killing FFmpeg: {e}")
            
        except Exception as e:
            logger.error(f"Error stopping FFmpeg: {e}")
        finally:
            ffmpeg_process = None
            streaming["running"] = False
            logger.info("FFmpeg stream stopped")
    
    def is_process_alive():
        """Check if FFmpeg process is still running."""
        if ffmpeg_process is None:
            return False
        return ffmpeg_process.poll() is None
    
    def stream_manager():
        """Manage FFmpeg stream lifecycle and handle configuration changes."""
        nonlocal server_config, config
        
        host = server_config["serverIP"]
        port = server_config["rtspPort"]
        
        dead_ticks = 0
        down_ticks = 0
        up_ticks = 0
        DEAD_THRESH = 3   # Require 3 consecutive process death detections (~30s)
        DOWN_THRESH = 3   # Require 3 consecutive connection failures before stop
        UP_THRESH = 2     # Require 2 consecutive successes before start
        
        while True:
            # Check for server configuration changes
            if birdCamera.waitForServerConfigChange(timeout=0.1):
                logger.info("Server configuration changed, reloading...")
                new_server_config = birdCamera.getCurrentServerConfig()
                
                # If server IP, port, or name changed, restart stream
                if (server_config["serverIP"] != new_server_config["serverIP"] or
                    server_config["rtspPort"] != new_server_config["rtspPort"] or
                    server_config["name"] != new_server_config["name"]):
                    
                    logger.info("Server connection details changed, restarting stream...")
                    with process_lock:
                        if streaming["running"]:
                            stop_stream()
                    
                    # Update server config
                    server_config = new_server_config
                    host = server_config["serverIP"]
                    port = server_config["rtspPort"]
                    
                    # Reset counters
                    dead_ticks = 0
                    down_ticks = 0
                    up_ticks = 0
                else:
                    server_config = new_server_config
            
            # Check for camera configuration changes
            if birdCamera.waitForConfigChange(timeout=0.1):
                logger.info("Camera configuration changed, reloading...")
                new_config = birdCamera.getCurrentConfig()
                
                # Check if video settings changed (requires restart)
                if (config.get("width") != new_config.get("width") or
                    config.get("height") != new_config.get("height") or
                    config.get("framerate") != new_config.get("framerate") or
                    config.get("bitrate") != new_config.get("bitrate") or
                    config.get("preset") != new_config.get("preset")):
                    
                    logger.info("Video settings changed, restarting stream...")
                    with process_lock:
                        if streaming["running"]:
                            stop_stream()
                    
                    config = new_config
                    
                    # Reset counters to trigger restart
                    dead_ticks = 0
                    down_ticks = 0
                    up_ticks = 0
                else:
                    config = new_config
            
            # Check if restart was explicitly requested
            if restart_stream.is_set():
                logger.info("Manual stream restart requested")
                with process_lock:
                    if streaming["running"]:
                        stop_stream()
                restart_stream.clear()
                dead_ticks = 0
                down_ticks = 0
                up_ticks = 0
            
            # Check process health and RTSP server availability
            up = _rtsp_up(host, port)
            
            with process_lock:
                # If streaming, check if process is still alive
                if streaming["running"]:
                    if not is_process_alive():
                        dead_ticks += 1
                        if dead_ticks >= DEAD_THRESH:
                            logger.warning(f"FFmpeg process dead for {dead_ticks} checks; restarting")
                            stop_stream()
                            dead_ticks = 0
                    else:
                        dead_ticks = 0
                
                # Check RTSP server availability
                if not up:
                    down_ticks += 1
                    up_ticks = 0
                    if streaming["running"] and down_ticks >= DOWN_THRESH:
                        logger.warning(f"RTSP server unreachable for {down_ticks} checks; stopping stream")
                        stop_stream()
                        down_ticks = 0
                else:
                    down_ticks = 0
                    if not streaming["running"]:
                        up_ticks += 1
                        if up_ticks >= UP_THRESH:
                            logger.info("RTSP server available, starting stream")
                            start_stream(config, server_config)
                            up_ticks = 0
                    else:
                        up_ticks = 0
            
            time.sleep(10)
    
    # Start stream manager thread
    logger.info("Starting stream manager")
    threading.Thread(target=stream_manager, daemon=True).start()
    
    # Keep main thread alive
    logger.info("Camera script running - press Ctrl+C to exit")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        with process_lock:
            if streaming["running"]:
                stop_stream()
        logger.info("Shutdown complete")

if __name__ == "__main__":
    runCamera()
