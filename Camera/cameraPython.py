import time
import os
import logging
import socket
import threading
import subprocess  # NEW

import numpy as np

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput, FfmpegOutput

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
    
    config = birdCamera.readConfig()
    
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

    def _get_ffmpeg_proc(out):
        # Picamera2 FfmpegOutput stores the Popen in different attrs across versions
        for name in ("process", "proc", "_process", "_ffmpeg", "_proc"):
            p = getattr(out, name, None)
            if p is not None:
                return p
        return None

    def _ffmpeg_dead(out) -> bool:
        if not out:
            return True
        p = _get_ffmpeg_proc(out)
        try:
            return (p is not None) and (p.poll() is not None)
        except Exception:
            return False

    def start_stream():
        nonlocal encoder, streamOutput
        try:
            logger.info("Starting RTSP encoder/output")
            enc = H264Encoder(2000000)
            try:
                enc.intra_period = 48  # friendlier for HLS recovery (best-effort)
            except Exception:
                pass
            out = FfmpegOutput(
                f'-f rtsp -rtsp_transport tcp rtsp://{config["serverIP"]}:{config["rtspPort"]}/{config["name"]}',
                audio=True,
                audio_codec="aac",
                audio_sync=config["audioDelay"],
            )
            # Prefer API with (encoder, output); fall back if needed
            try:
                Picamera2.start_encoder  # probe existence
                picam2.start_encoder(enc, out)
            except TypeError:
                # older signatures
                enc.output = out
                picam2.start_encoder()
            except AttributeError:
                # very old API; use output assignment
                enc.output = out
                picam2.start_encoder()

            encoder = enc
            streamOutput = out
            streaming["running"] = True
            proc = _get_ffmpeg_proc(out)
            if proc:
                logger.info(f"ffmpeg pid={getattr(proc,'pid',None)} started")
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
        host = config["serverIP"]
        port = config["rtspPort"]
        while True:
            up = _rtsp_up(host, port)
            with encoder_lock:
                # If ffmpeg exited (Broken pipe or server-side EOF), restart
                if streaming["running"] and _ffmpeg_dead(streamOutput):
                    logger.warning("ffmpeg output exited; restarting stream")
                    stop_stream()
                if up and not streaming["running"]:
                    start_stream()
                elif (not up) and streaming["running"]:
                    logger.warning("RTSP server down; stopping encoder to avoid broken pipe")
                    stop_stream()
            time.sleep(2)

    # Start camera and manager; encoder will start when RTSP is reachable
    picam2.start()
    threading.Thread(target=stream_manager, daemon=True).start()

    w, h = lsize

    prev = picam2.capture_buffer("lores")
    prev = prev[:w * h].reshape(h, w).astype(np.int16)

    bwMode = False # greyscale mode on/off
    currBrightness = 0
    skipNFrames = 10 # skip the first frames to avoid recording on startup

    while True:

        # capture new preview and reshape
        cur = picam2.capture_buffer("lores")
        cur = cur[:w * h].reshape(h, w).astype(np.int16)
        #calculate current brightness
        currBrightness = np.square(cur).mean()

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
