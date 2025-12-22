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
            logger.info("="*60)
            logger.info(result.stdout)
            logger.info("="*60)
            
            # Parse and highlight YUYV/YUY2 formats
            lines = result.stdout.split('\n')
            for i, line in enumerate(lines):
                if 'YUYV' in line or 'YUY2' in line or 'YUYV 4:2:2' in line:
                    logger.info("\n>>> YUYV format found, available resolutions:")
                    for j in range(i+1, min(i+30, len(lines))):
                        if lines[j].strip() and not lines[j].strip().startswith('['):
                            logger.info(lines[j])
                        if j < len(lines)-1 and lines[j+1].strip().startswith('['):
                            break
        else:
            logger.warning(f"Could not detect camera capabilities: {result.stderr}")
            
    except FileNotFoundError:
        logger.warning("v4l2-ctl not found, install v4l-utils to detect camera capabilities")
    except Exception as e:
        logger.error(f"Error detecting camera capabilities: {e}")

def set_camera_format(device, width, height, framerate, pixel_format='YUYV'):
    """Try to set camera format using v4l2-ctl before FFmpeg starts."""
    try:
        logger.info(f"Attempting to set camera format: {width}x{height} @ {framerate}fps, format={pixel_format}")
        
        # Set pixel format
        result = subprocess.run(
            ['v4l2-ctl', '--device', device, '--set-fmt-video',
             f'width={width},height={height},pixelformat={pixel_format}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            logger.info(f"Format set result: {result.stdout}")
        else:
            logger.warning(f"Could not set format: {result.stderr}")
        
        # Set framerate
        result = subprocess.run(
            ['v4l2-ctl', '--device', device, '--set-parm', f'{framerate}'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            logger.info(f"Framerate set result: {result.stdout}")
        else:
            logger.warning(f"Could not set framerate: {result.stderr}")
            
    except FileNotFoundError:
        logger.warning("v4l2-ctl not found, skipping pre-configuration")
    except Exception as e:
        logger.error(f"Error setting camera format: {e}")

def detect_audio_capabilities(device):
    """Detect audio device capabilities using arecord."""
    try:
        logger.info(f"Detecting audio capabilities for {device}:")
        
        # Get device info
        result = subprocess.run(
            ['arecord', '-L'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            logger.info("Available ALSA devices:")
            logger.info(result.stdout)
        
        # Try to get hardware parameters
        result = subprocess.run(
            ['arecord', '-D', device, '--dump-hw-params', '-d', '1'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            logger.info(f"Hardware parameters for {device}:")
            logger.info(result.stderr)  # arecord outputs to stderr
        else:
            logger.warning(f"Could not get hardware parameters: {result.stderr}")
            
    except FileNotFoundError:
        logger.warning("arecord not found, skipping audio detection")
    except Exception as e:
        logger.error(f"Error detecting audio capabilities: {e}")

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
    
    # If audio is enabled in config, detect audio capabilities too
    if config.get("enableAudio"):
        audio_device = config.get("audioDevice", "hw:0,0")
        logger.info(f"Audio enabled, detecting capabilities for {audio_device}")
        detect_audio_capabilities(audio_device)

    # State management
    ffmpeg_process = None
    process_lock = threading.Lock()
    streaming = {"running": False}
    restart_stream = threading.Event()
    
    # Video device - adjust if needed
    video_device = config.get("videoDevice", "/dev/video0")
    
    def build_ffmpeg_command(cfg, srv_cfg):
        """Build FFmpeg command based on current configuration."""
        width = cfg.get("width", 1920)
        height = cfg.get("height", 1080)
        framerate = cfg.get("framerate", 25)
        bitrate = cfg.get("bitrate", "3000k")
        preset = cfg.get("preset", "ultrafast")
        input_format = cfg.get("inputFormat", "mjpeg")  # mjpeg or yuyv422
        enable_audio = cfg.get("enableAudio", False)
        audio_device = cfg.get("audioDevice", "hw:0,0")  # ALSA device for audio
        audio_channels = cfg.get("audioChannels", 1)  # 1 for mono, 2 for stereo
        audio_sample_rate = cfg.get("audioSampleRate", 16000)
        audio_buffer_size = cfg.get("audioBufferSize", 4096)  # ALSA buffer size
        
        rtsp_url = f'rtsp://{srv_cfg["serverIP"]}:{srv_cfg["rtspPort"]}/{srv_cfg["name"]}'
        
        # Pre-configure camera using v4l2-ctl for better control
        pixel_fmt = 'YUYV' if 'yuyv' in input_format.lower() else 'MJPG'
        set_camera_format(video_device, width, height, framerate, pixel_fmt)
        
        # Build FFmpeg command
        cmd = [
            'ffmpeg',
            '-f', 'v4l2',
            '-input_format', input_format,
            '-video_size', f'{width}x{height}',
            '-framerate', str(framerate),
            '-thread_queue_size', '512',  # Video input buffer
            '-i', video_device,
        ]
        
        # Add audio input if enabled
        if enable_audio:
            # Use plughw instead of hw for better compatibility
            alsa_device = audio_device
            if audio_device.startswith('hw:'):
                # Try plughw for automatic sample rate/format conversion
                alsa_device = 'plughw:' + audio_device[3:]
                logger.info(f"Using {alsa_device} (plugin device) for better compatibility")
            
            cmd.extend([
                '-f', 'alsa',
                '-thread_queue_size', '2048',  # Larger buffer for stability
                '-channels', str(audio_channels),
                '-sample_rate', str(audio_sample_rate),
                '-i', alsa_device,
            ])
        
        # Video encoding options
        cmd.extend([
            '-c:v', 'libx264',
            '-preset', preset,
            '-tune', 'zerolatency',  # Minimize encoding delay
            '-b:v', bitrate,
            '-maxrate', bitrate,  # Cap bitrate
            '-bufsize', str(int(bitrate.replace('k', '')) * 2) + 'k',  # Buffer size = 2x bitrate
            '-pix_fmt', 'yuv420p',
            '-color_range', 'tv',  # Fix color range warning
            '-colorspace', 'bt709',
            '-g', str(framerate * 2),  # Keyframe every 2 seconds
            '-flags', '+low_delay',  # Low delay mode
            '-fflags', '+nobuffer',  # No buffering
        ])
        
        # Audio encoding options if enabled
        if enable_audio:
            cmd.extend([
                '-c:a', 'aac',
                '-b:a', '32k' if audio_channels == 1 else '64k',  # Lower bitrate for mono
                # Audio filters to reduce noise
                '-af', 'highpass=f=200,lowpass=f=3000,volume=2',  # Filter out rumble/hiss, boost volume
            ])
        
        # RTSP output
        cmd.extend([
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',
            rtsp_url
        ])
        
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
                    config.get("preset") != new_config.get("preset") or
                    config.get("inputFormat") != new_config.get("inputFormat") or
                    config.get("enableAudio") != new_config.get("enableAudio") or
                    config.get("audioDevice") != new_config.get("audioDevice") or
                    config.get("audioChannels") != new_config.get("audioChannels") or
                    config.get("audioSampleRate") != new_config.get("audioSampleRate") or
                    config.get("audioBufferSize") != new_config.get("audioBufferSize")):
                    
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
