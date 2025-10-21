import time
import os
import logging
import socket
import threading
import subprocess
import signal          # NEW
import re              # NEW

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

# NEW: best-effort ALSA device discovery (card,device) -> "hw:X,Y"
def _detect_alsa_device() -> str | None:
    try:
        out = subprocess.check_output(["arecord", "-l"], text=True)
    except Exception:
        return None
    m = re.findall(r"card\s+(\d+).+device\s+(\d+):", out, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    card, dev = m[0]
    return f"hw:{card},{dev}"

AUDIO_CH = int(os.environ.get("BIRBCAM_AUDIO_CHANNELS", "1"))
AUDIO_SR = int(os.environ.get("BIRBCAM_AUDIO_RATE", "48000"))

def runCamera():
    signal.signal(signal.SIGHUP, signal.SIG_IGN)

    requested_dev = os.environ.get("BIRBCAM_AUDIO_DEV")  # e.g. "plughw:0,0"
    alsa_dev = requested_dev or _detect_alsa_device()
    use_audio_default = os.environ.get("BIRBCAM_ENABLE_AUDIO", "1") not in ("0", "false", "False")

    if alsa_dev and use_audio_default:
        os.environ.pop("PULSE_SERVER", None)
        logger.info(f"Using ALSA audio device: {alsa_dev}")
    elif use_audio_default:
        logger.warning("No ALSA capture device detected; will try audio, may fall back to video-only")
    else:
        logger.info("Audio disabled by env (BIRBCAM_ENABLE_AUDIO=0)")

    logging.basicConfig(
        filename='birb.log',
        format='%(asctime)s %(levelname)s %(module)s %(funcName)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

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
    streaming = {
        "running": False,
        "use_audio": use_audio_default,   # track effective audio usage
        "audio_failures": 0               # count consecutive audio-related exits
    }

    rtsp_url = f'rtsp://{config["serverIP"]}:{config["rtspPort"]}/{config["name"]}'

    def _find_ffmpeg(target_url: str):
        """Return (pid, cmdline) for ffmpeg process that matches the stream URL."""
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
                        return int(pid), cmd
                except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
                    continue
        except Exception:
            pass
        return None, None

    def _ffmpeg_dead() -> bool:
        pid, _ = _find_ffmpeg(rtsp_url)
        return pid is None

    def start_stream():
        nonlocal encoder, streamOutput
        try:
            logger.info("Starting RTSP encoder/output")
            enc = H264Encoder(2000000)
            try:
                enc.intra_period = 48
            except Exception:
                pass

            # Choose device string for ALSA (prefer plughw for format conversion)
            dev = None
            if streaming["use_audio"] and alsa_dev:
                dev = alsa_dev
                if not dev.startswith(("hw:", "plughw:")):
                    dev = f"hw:{dev}"
                if dev.startswith("hw:"):
                    dev = "plughw" + dev[2:]

            try:
                if streaming["use_audio"] and dev:
                    out = FfmpegOutput(
                        f'-f rtsp -rtsp_transport tcp {rtsp_url}',
                        audio=True,
                        audio_device=dev,
                        audio_channels=AUDIO_CH,
                        audio_samplerate=AUDIO_SR,
                        audio_codec="aac"
                    )
                elif streaming["use_audio"]:
                    # May default to Pulse in older Picamera2; we will detect and fallback if it fails
                    out = FfmpegOutput(
                        f'-f rtsp -rtsp_transport tcp {rtsp_url}',
                        audio=True,
                        audio_channels=AUDIO_CH,
                        audio_samplerate=AUDIO_SR,
                        audio_codec="aac"
                    )
                else:
                    out = FfmpegOutput(
                        f'-f rtsp -rtsp_transport tcp {rtsp_url}',
                        audio=False
                    )
            except TypeError:
                # Older Picamera2 without audio kwargs
                params = f'-f rtsp -rtsp_transport tcp {rtsp_url}'
                out = FfmpegOutput(params, audio=streaming["use_audio"])

            # Start encoder with output
            try:
                Picamera2.start_encoder  # probe
                picam2.start_encoder(enc, out)
            except TypeError:
                enc.output = out
                picam2.start_encoder()
            except AttributeError:
                enc.output = out
                picam2.start_encoder()

            # allow ffmpeg to spawn
            time.sleep(0.8)
            pid, cmd = _find_ffmpeg(rtsp_url)
            if pid:
                logger.info(f"ffmpeg pid={pid} started; cmd='{cmd}'")
                # Warn if Pulse backend is used (will fail without session)
                if " -f pulse " in f" {cmd} " or " pulse " in f" {cmd} ":
                    logger.warning("ffmpeg is using PulseAudio backend; if it dies on logout, set BIRBCAM_AUDIO_DEV=plughw:X,Y")
            else:
                logger.warning("ffmpeg pid not found (watchdog will verify liveness)")

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
        host = config["serverIP"]
        port = config["rtspPort"]
        dead_ticks = 0
        down_ticks = 0
        up_ticks = 0
        audio_flips = 0
        DEAD_THRESH = 3
        DOWN_THRESH = 3
        UP_THRESH = 2
        while True:
            up = _rtsp_up(host, port)
            with encoder_lock:
                if streaming["running"]:
                    if _ffmpeg_dead():
                        dead_ticks += 1
                        if dead_ticks >= DEAD_THRESH:
                            # Inspect last known cmd; if Pulse was used and no Pulse server, disable audio
                            _, cmd = _find_ffmpeg(rtsp_url)  # usually None if dead, but try
                            using_pulse = bool(cmd and (" -f pulse " in f" {cmd} " or " pulse " in f" {cmd} "))
                            logger.warning("ffmpeg missing for %d checks; restarting stream", dead_ticks)
                            stop_stream()
                            if streaming["use_audio"] and (using_pulse or not alsa_dev):
                                streaming["use_audio"] = False
                                audio_flips += 1
                                logger.warning("Disabling audio (fallback to video-only). Set BIRBCAM_AUDIO_DEV=plughw:X,Y to re-enable.")
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

    logger.info(f"Audio config: use_audio={streaming["use_audio"]}, alsa_dev={alsa_dev}, ch={AUDIO_CH}, sr={AUDIO_SR}")

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
