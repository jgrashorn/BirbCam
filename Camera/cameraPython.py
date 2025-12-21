import time
import os
import logging
import socket
import threading
import subprocess

import numpy as np

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

import libcamera
import birdCamera

logger = logging.getLogger(__name__)

def _rtsp_up(host, port, timeout=1.0):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False

def runCamera():

    os.environ['PULSE_SERVER'] = "/run/user/1000/pulse/native"

    logging.basicConfig(
                        filename='birb.log',
                        format='%(asctime)s %(levelname)s %(module)s %(funcName)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    # Load server configuration first
    server_config = birdCamera.readServerConfig()
    
    # Load camera configuration
    config = birdCamera.readConfig()
    
    # Start settings server
    settings_port = server_config.get("settingsPort", 5005)
    birdCamera.startSettingsServer(settings_port)
    
    lsize = (180, 120) # size of internal preview for motion detection (smol bc fast)
    msize = (config["width"], config["height"]) # size of recording from config.txt
    picam2 = Picamera2()

    video_config = picam2.create_video_configuration(main={"size": msize, "format": "YUV420"}, # format of recording
                                                    lores={"size": lsize, "format": "YUV420"}, # format of preview
                                                    controls={"ColourGains": (config["colorOffset_red"], config["colorOffset_blue"])} # color correction for IR-cams, (r, b)
                                                    )
    # transforms if camera is not oriented right side up
    video_config["transform"] = libcamera.Transform(hflip=0, vflip=0)
    
    picam2.configure(video_config)

    # --- self-healing RTSP publisher ---
    encoder_lock = threading.Lock()
    encoder = None
    streamOutput = None
    streaming = {"running": False}

    rtsp_url = f'rtsp://{server_config["serverIP"]}:{server_config["rtspPort"]}/{server_config["name"]}'

    # Replace pgrep with a /proc scan (more reliable, no shell)
    def _find_ffmpeg_pid(target_url: str):
        try:
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    with open(f"/proc/{pid}/cmdline", "rb") as f:
                        raw = f.read()
                    if not raw:
                        continue
                    parts = raw.split(b"\x00")
                    exe = parts[0].decode("utf-8", "ignore")
                    if "ffmpeg" not in exe:
                        continue
                    cmd = " ".join(p.decode("utf-8", "ignore") for p in parts if p)
                    if target_url in cmd:
                        return int(pid)
                except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
                    continue
        except Exception:
            pass
        return None

    def _ffmpeg_dead() -> bool:
        return _find_ffmpeg_pid(rtsp_url) is None

    def start_stream():
        nonlocal encoder, streamOutput, rtsp_url
        try:
            logger.info(f"Starting RTSP encoder/output to {rtsp_url}")
            enc = H264Encoder(2000000)
            try:
                enc.intra_period = 48  # friendlier for HLS recovery (best-effort)
            except Exception:
                pass
            out = FfmpegOutput(
                f'-f rtsp -rtsp_transport tcp {rtsp_url}',
                audio=True
            )
            # Start encoder with output
            try:
                Picamera2.start_encoder  # probe existence
                picam2.start_encoder(enc, out)
            except TypeError:
                enc.output = out
                picam2.start_encoder()
            except AttributeError:
                enc.output = out
                picam2.start_encoder()

            # give ffmpeg a moment to spawn, then locate pid
            time.sleep(0.5)
            pid = _find_ffmpeg_pid(rtsp_url)
            if pid:
                logger.info(f"ffmpeg pid={pid} started")
            else:
                logger.warning("ffmpeg pid not found (debounced monitor will verify liveness)")

            encoder = enc
            streamOutput = out
            streaming["running"] = True
        except Exception as e:
            logger.error(f"Failed to start stream: {e}", exc_info=True)
            streaming["running"] = False

    def stop_stream():
        nonlocal encoder, streamOutput
        try:
            logger.info("Stopping RTSP encoder/output")
            try:
                picam2.stop_encoder()
            except Exception:
                pass
            try:
                if hasattr(streamOutput, "stop"):
                    streamOutput.stop()
            except Exception:
                pass
            try:
                if hasattr(encoder, "close"):
                    encoder.close()
            except Exception:
                pass
        finally:
            encoder = None
            streamOutput = None
            streaming["running"] = False

    def stream_manager():
        nonlocal rtsp_url, server_config
        host = server_config["serverIP"]
        port = server_config["rtspPort"]
        dead_ticks = 0
        down_ticks = 0
        up_ticks = 0
        DEAD_THRESH = 3   # require 3 consecutive misses (~6s) before restart
        DOWN_THRESH = 3   # require 3 consecutive connect failures before stop
        UP_THRESH = 2     # require 2 consecutive successes before start
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
                    with encoder_lock:
                        if streaming["running"]:
                            stop_stream()
                    
                    # Update server config and rtsp_url
                    server_config = new_server_config
                    rtsp_url = f'rtsp://{server_config["serverIP"]}:{server_config["rtspPort"]}/{server_config["name"]}'
                    host = server_config["serverIP"]
                    port = server_config["rtspPort"]
                    
                    logger.info(f"New RTSP URL: {rtsp_url}")
                    
                    # Reset counters
                    dead_ticks = 0
                    down_ticks = 0
                    up_ticks = 0
                else:
                    server_config = new_server_config
            
            up = _rtsp_up(host, port)
            with encoder_lock:
                if streaming["running"]:
                    if _ffmpeg_dead():
                        dead_ticks += 1
                        if dead_ticks >= DEAD_THRESH:
                            logger.warning("ffmpeg missing for %d checks; restarting stream", dead_ticks)
                            stop_stream()
                            dead_ticks = 0
                    else:
                        dead_ticks = 0

                if not up:
                    down_ticks += 1
                    up_ticks = 0
                    if streaming["running"] and down_ticks >= DOWN_THRESH:
                        logger.warning("RTSP not reachable for %d checks; stopping encoder", down_ticks)
                        stop_stream()
                        down_ticks = 0
                else:
                    down_ticks = 0
                    if not streaming["running"]:
                        up_ticks += 1
                        if up_ticks >= UP_THRESH:
                            start_stream()
                            up_ticks = 0
                    else:
                        up_ticks = 0
            time.sleep(10)

    # Start camera and manager; encoder will start when RTSP is reachable
    picam2.start()
    threading.Thread(target=stream_manager, daemon=True).start()

    w, h = lsize

    prev = picam2.capture_buffer("lores")
    prev = prev[:w * h].reshape(h, w).astype(np.int16)

    bwMode = False # greyscale mode on/off
    picam2.set_controls({"Saturation": 1.0})
    currBrightness = 0
    skipNFrames = 10 # skip the first frames to avoid recording on startup

    while True:
        # Check for configuration changes
        if birdCamera.waitForConfigChange(timeout=0.1):  # Non-blocking check
            logger.info("Configuration changed, reloading...")
            new_config = birdCamera.getCurrentConfig()
            
            # Update color gains if they changed
            if (config["colorOffset_red"] != new_config["colorOffset_red"] or
                config["colorOffset_blue"] != new_config["colorOffset_blue"]):
                try:
                    picam2.set_controls({
                        "ColourGains": (
                            new_config["colorOffset_red"], 
                            new_config["colorOffset_blue"]
                        )
                    })
                    logger.info("Updated color gains")
                except Exception as e:
                    logger.error(f"Failed to update color gains: {e}")
            
            # Update configuration reference
            config.update(new_config)
            skipNFrames = config["skippedFramesAfterChange"]  # Reset frames after config change
            logger.info(config)
            
        # capture new preview and reshape
        cur = picam2.capture_buffer("lores")
        cur = cur[:w * h].reshape(h, w).astype(np.float32)
        #calculate current brightness
        currBrightness = np.square(cur).mean()

        # logger.info(f"Image: {cur}, Current brightness: {currBrightness}, bwMode: {bwMode}, skipNFrames: {skipNFrames}")

        # skip some frames, e.g. if mode was changed
        if skipNFrames > 0: 
            skipNFrames -= 1

        else:
            # switch mode in case brightness reached threshold, then skip some frames
            if bwMode and currBrightness > config["clrSwitchingSensitivity"]:
                logger.info("switching mode to color")
                picam2.set_controls({"Saturation": 1.0})
                bwMode = False
                skipNFrames = config["skippedFramesAfterChange"]

            elif not bwMode and currBrightness < config["bwSwitchingSensitivity"]:
                logger.info("switching mode to greyscale")
                picam2.set_controls({"Saturation": 0.0})
                bwMode = True
                skipNFrames = config["skippedFramesAfterChange"]

        prev = cur # overwrite previous frame with current one

if __name__ == "__main__":
    runCamera()
