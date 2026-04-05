import time
import os
import logging
import socket
import threading
import subprocess

import numpy as np

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, MJPEGEncoder
from picamera2.outputs import PyavOutput

import libcamera
import birdCamera

logger = logging.getLogger(__name__)

def _rtsp_up(host, port, timeout=1.0):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False

def _detect_audio_source(preferred_source=None):
    """Return a usable PulseAudio capture source name, or None if no physical mic is available."""
    try:
        pulse_socket = os.environ.get('PULSE_SERVER', '/run/user/1000/pulse/native')
        if pulse_socket.startswith('/') and not os.path.exists(pulse_socket):
            logger.info("PulseAudio socket not found, disabling audio")
            return None

        info_result = subprocess.run(
            ['pactl', 'info'],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if info_result.returncode != 0:
            logger.info("PulseAudio not responding, disabling audio")
            return None

        default_source = None
        default_result = subprocess.run(
            ['pactl', 'get-default-source'],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if default_result.returncode == 0:
            default_source = default_result.stdout.strip()

        sources_result = subprocess.run(
            ['pactl', 'list', 'short', 'sources'],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if sources_result.returncode != 0:
            logger.info("Could not list PulseAudio sources, disabling audio")
            return None

        sources = []
        for line in sources_result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                sources.append({
                    "name": parts[1].strip(),
                    "driver": parts[2].strip() if len(parts) >= 3 else "",
                    "state": parts[-1].strip(),
                })

        physical_sources = [
            source for source in sources
            if not source["name"].endswith(".monitor")
            and "monitor" not in source["name"].lower()
        ]

        if preferred_source:
            for source in physical_sources:
                if source["name"] == preferred_source:
                    logger.info(f"Using configured audio source: {preferred_source} ({source['state']})")
                    return preferred_source
            logger.warning(
                f"Configured audio source '{preferred_source}' not found; falling back to auto-detection"
            )

        if default_source:
            for source in physical_sources:
                if source["name"] == default_source:
                    logger.info(f"Using default PulseAudio source: {default_source} ({source['state']})")
                    return default_source

        preferred_keywords = ("usb", "mic", "input")
        for source in physical_sources:
            driver_text = f"{source.get('driver', '')} {source['name']}".lower()
            if any(keyword in driver_text for keyword in preferred_keywords):
                logger.info(f"Using preferred audio source: {source['name']} ({source['state']})")
                return source["name"]

        if physical_sources:
            fallback = physical_sources[0]
            logger.info(f"Using detected audio source: {fallback['name']} ({fallback['state']})")
            return fallback["name"]

        logger.info(
            f"No physical PulseAudio input sources found (available sources: {[s['name'] for s in sources]}), "
            "disabling audio"
        )
        return None

    except FileNotFoundError:
        logger.info("pactl command not found, disabling audio")
        return None
    except subprocess.TimeoutExpired:
        logger.info("PulseAudio query timeout, disabling audio")
        return None
    except Exception as e:
        logger.warning(f"Audio source detection failed: {e}, disabling audio")
        return None

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

    encoder_lock = threading.Lock()
    camera_lock = threading.RLock()
    encoder = None
    streamOutput = None
    streaming = {"running": False}
    stream_restart_requested = threading.Event()
    preferred_sensor = None

    props = picam2.camera_properties
    sensor_modes = list(getattr(picam2, "sensor_modes", []) or [])
    logger.info(f"camera properties: {props}")
    logger.info(f"available sensor modes: {sensor_modes}")

    def choose_main_format(picam2):
        model = picam2.camera_properties["Model"]

        if "imx477" in model: # Raspberry Pi HQ Camera
            return "RGB8888"
        elif "imx708" in model: # Raspberry Pi Camera Module v3 (all variants)
            return "YUV420"
        else:
            return "YUV420"  # prefer YUV for most cameras

    MAIN_FORMAT = choose_main_format(picam2)
    logger.info(f"choosing {MAIN_FORMAT}")

    def build_transform(camera_config):
        return libcamera.Transform(
            hflip=bool(camera_config.get("hflip", 0)),
            vflip=bool(camera_config.get("vflip", 0))
        )

    def build_frame_duration_limits(camera_config):
        framerate = max(float(camera_config.get("framerate", 30) or 30), 1.0)
        frame_duration_us = int(1_000_000 / framerate)
        return (frame_duration_us, frame_duration_us)

    def choose_sensor_config(camera_config):
        requested_w = int(camera_config["width"])
        requested_h = int(camera_config["height"])
        requested_fps = float(camera_config.get("framerate", 30) or 30)

        ranked_modes = []
        for mode in sensor_modes:
            size = tuple(mode.get("size") or ())
            if len(size) != 2:
                continue

            fps = float(mode.get("fps", 0) or 0)
            fits_size = size[0] >= requested_w and size[1] >= requested_h
            fits_fps = fps >= (requested_fps - 0.5)
            score = (
                0 if fits_size else 1,
                0 if fits_fps else 1,
                -fps,
                size[0] * size[1],
                -int(mode.get("bit_depth", 0) or 0),
            )
            ranked_modes.append((score, mode))

        if not ranked_modes:
            logger.warning("No sensor modes reported; falling back to automatic mode selection")
            return None

        chosen = min(ranked_modes, key=lambda item: item[0])[1]
        chosen_size = tuple(chosen.get("size") or ())
        if len(chosen_size) != 2:
            return None

        if chosen_size[0] < requested_w or chosen_size[1] < requested_h:
            logger.warning(
                f"No sensor mode fully covers {requested_w}x{requested_h}; using automatic mode selection"
            )
            return None

        sensor_config = {"output_size": chosen_size}
        bit_depth = chosen.get("bit_depth")
        if bit_depth is not None:
            sensor_config["bit_depth"] = int(bit_depth)

        logger.info(
            f"Selected sensor mode for {requested_w}x{requested_h} @ {requested_fps} FPS: "
            f"size={chosen_size}, fps={chosen.get('fps')}, bit_depth={chosen.get('bit_depth')}"
        )
        return sensor_config

    def build_video_config(camera_config):
        controls = {
            "FrameDurationLimits": build_frame_duration_limits(camera_config),
        }

        if camera_config.get("awbEnable", False):
            controls["AwbEnable"] = True
        else:
            controls["AwbEnable"] = False
            controls["ColourGains"] = (
                camera_config["colorOffset_red"],
                camera_config["colorOffset_blue"],
            )

        create_kwargs = {
            "main": {"size": (camera_config["width"], camera_config["height"]), "format": MAIN_FORMAT},
            "lores": {"size": lsize, "format": "YUV420"},
            "controls": controls,
            "transform": build_transform(camera_config),
        }

        selected_sensor = choose_sensor_config(camera_config) or preferred_sensor
        if selected_sensor:
            output_size = selected_sensor.get("output_size") or selected_sensor.get("size")
            if output_size and (
                output_size[0] < camera_config["width"] or
                output_size[1] < camera_config["height"]
            ):
                logger.info(
                    "Requested resolution exceeds selected sensor mode; allowing libcamera to reselect"
                )
            else:
                create_kwargs["sensor"] = selected_sensor
                logger.info(f"Using explicit sensor configuration: {selected_sensor}")

        try:
            return picam2.create_video_configuration(**create_kwargs)
        except TypeError as e:
            if "sensor" in create_kwargs:
                logger.warning(
                    f"Sensor pinning unsupported by this Picamera2 build ({e}); using default mode selection"
                )
                create_kwargs.pop("sensor", None)
                return picam2.create_video_configuration(**create_kwargs)
            raise

    def apply_runtime_camera_settings(camera_config):
        try:
            with camera_lock:
                picam2.set_controls({
                    "FrameDurationLimits": build_frame_duration_limits(camera_config)
                })
            logger.info(f"Applied frame duration limits for {camera_config.get('framerate', 30)} FPS")
        except Exception as e:
            logger.error(f"Failed to apply frame duration limits: {e}")

        try:
            af_mode = 2 if camera_config.get("autofocus", False) else 0
            with camera_lock:
                picam2.set_controls({"AfMode": af_mode, "AfTrigger": 0})
            logger.info("Autofocus enabled" if af_mode == 2 else "Autofocus disabled")
        except Exception as e:
            logger.error(f"Failed to apply autofocus settings: {e}")

    def configure_camera(camera_config):
        nonlocal msize, preferred_sensor
        msize = (camera_config["width"], camera_config["height"])
        video_config = build_video_config(camera_config)
        with camera_lock:
            picam2.configure(video_config)
            active_config = picam2.camera_configuration()

        sensor_config = active_config.get("sensor") if isinstance(active_config, dict) else None
        if sensor_config:
            preferred_sensor = dict(sensor_config)
            logger.info(f"Pinned active sensor configuration for future rebuilds: {preferred_sensor}")

        logger.info(f"Active camera configuration: {active_config}")
        logger.info(
            f"Configured camera: {msize[0]}x{msize[1]} @ {camera_config.get('framerate', 30)} FPS, "
            f"hflip={camera_config.get('hflip', 0)}, vflip={camera_config.get('vflip', 0)}"
        )

    def log_camera_timing(context):
        try:
            with camera_lock:
                metadata = picam2.capture_metadata()

            frame_duration = metadata.get("FrameDuration")
            approx_fps = (1_000_000 / frame_duration) if frame_duration else None
            logger.info(
                f"{context} metadata: FrameDuration={frame_duration}, "
                f"approx_fps={approx_fps:.2f} "
                f"ExposureTime={metadata.get('ExposureTime')}, "
                f"AnalogueGain={metadata.get('AnalogueGain')}, "
                f"AeLocked={metadata.get('AeLocked')}"
                if approx_fps is not None else f"{context} metadata: {metadata}"
            )
        except Exception as e:
            logger.warning(f"Failed to read camera metadata after reconfiguration: {e}")

    def reconfigure_camera_pipeline(new_config):
        logger.info(
            f"Synchronously rebuilding camera pipeline for {new_config['width']}x{new_config['height']} "
            f"at {new_config.get('framerate', 30)} FPS"
        )
        with encoder_lock:
            if streaming["running"]:
                stop_stream()

            with camera_lock:
                try:
                    logger.info("Stopping camera...")
                    picam2.stop()
                except Exception as e:
                    logger.error(f"Error stopping camera: {e}")

                configure_camera(new_config)
                picam2.start()

            apply_runtime_camera_settings(new_config)
            log_camera_timing("Post-reconfigure")

        config.update(new_config)
        logger.info("Camera reconfigured and restarted")

    configure_camera(config)

    min_exp, max_exp, default_exp = picam2.camera_controls["ExposureTime"]
    min_frameduration, max_frameduration, default_frameduration = picam2.camera_controls["FrameDurationLimits"]
    logger.info(f"ExposureTime range: {min_exp} - {max_exp}, default: {default_exp}")
    logger.info(f"FrameDurationLimits range: {min_frameduration} - {max_frameduration}, default: {default_frameduration}")

    # --- self-healing RTSP publisher ---
    rtsp_url = f'rtsp://{server_config["serverIP"]}:{server_config["rtspPort"]}/{server_config["name"]}'

    def _stream_output_dead() -> bool:
        return bool(streamOutput is not None and getattr(streamOutput, "_container", None) is None)

    def _on_stream_output_error(error):
        logger.warning(f"RTSP output error from PyAV: {error}")
        stream_restart_requested.set()

    def start_stream():
        nonlocal encoder, streamOutput, rtsp_url
        try:
            logger.info(f"Starting RTSP encoder/output to {rtsp_url}")

            target_fps = max(float(config.get("framerate", 30) or 30), 1.0)
            encoder_type = config.get("encoderType", "h264").lower()
            logger.info(f"Selected encoder type: {encoder_type}, target FPS: {target_fps}")

            if encoder_type == "mjpeg":
                logger.info("Using MJPEG encoder")
                enc = MJPEGEncoder()
                try:
                    enc.framerate = target_fps
                except Exception:
                    pass
            else:
                logger.info("Using H264 encoder")
                enc = H264Encoder(
                    2000000,
                    repeat=True,
                    iperiod=max(int(target_fps * 2), 1),
                    framerate=target_fps,
                    enable_sps_framerate=True,
                )

            # Detect actual microphone availability (not just whether PulseAudio is running)
            audio_enabled = bool(config.get("audioEnabled", True))
            configured_audio_source = config.get("audioDevice") or None
            audio_source = _detect_audio_source(configured_audio_source) if audio_enabled else None
            audio_available = audio_source is not None
            audio_delay = float(config.get("audioDelay", 0.0) or 0.0)
            audio_samplerate = int(config.get("audioSampleRate", 16000) or 16000)
            audio_bitrate = int(config.get("audioBitrate", 32000) or 32000)
            audio_codec = config.get("audioCodec", "aac") or "aac"
            audio_filter = config.get("audioFilter", "") or ""

            enc.audio = audio_available
            if audio_available:
                enc.audio_input = {
                    "file": audio_source,
                    "format": "pulse",
                }
                enc.audio_output = {
                    "codec_name": audio_codec,
                    "rate": audio_samplerate,
                    "bit_rate": audio_bitrate,
                }
                enc.audio_sync = int(audio_delay * 1_000_000)
                logger.info(f"Audio enabled using source: {audio_source}")
                if audio_filter:
                    logger.info("`audioFilter` is ignored with `PyavOutput`; PyAV handles audio muxing directly")
            elif not audio_enabled:
                logger.info("Audio explicitly disabled in config")
            else:
                logger.info("No usable microphone source detected; starting stream without audio")

            output_options = {
                "rtsp_transport": "tcp",
                "muxdelay": "0.0",
                "muxpreload": "0.0",
            }
            out = PyavOutput(rtsp_url, format="rtsp", options=output_options)
            out.error_callback = _on_stream_output_error

            # Start encoder with output
            with camera_lock:
                try:
                    Picamera2.start_encoder  # probe existence
                    picam2.start_encoder(enc, out)
                except TypeError:
                    enc.output = out
                    picam2.start_encoder()
                except AttributeError:
                    enc.output = out
                    picam2.start_encoder()

            try:
                if hasattr(enc, "force_key_frame"):
                    enc.force_key_frame()
            except Exception:
                pass

            logger.info(
                f"PyAV output started with nominal framerate={getattr(enc, 'framerate', None)}, "
                f"audio={audio_available}, audio_source={audio_source}, audio_sync_s={audio_delay}, "
                f"audio_samplerate={audio_samplerate}, audio_bitrate={audio_bitrate}, "
                f"audio_codec={audio_codec}, rtsp_options={output_options}"
            )

            encoder = enc
            streamOutput = out
            streaming["running"] = True

            log_camera_timing("After stream start")
        except Exception as e:
            logger.error(f"Failed to start stream: {e}", exc_info=True)
            streaming["running"] = False

    def stop_stream():
        nonlocal encoder, streamOutput
        try:
            logger.info("Stopping RTSP encoder/output")
            with camera_lock:
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
            logger.info("RTSP encoder/output stopped")

    def stream_manager():
        nonlocal rtsp_url, server_config, config, msize
        host = server_config["serverIP"]
        port = server_config["rtspPort"]
        dead_ticks = 0
        down_ticks = 0
        up_ticks = 0
        DEAD_THRESH = 3   # require 3 consecutive misses (~3s) before restart
        DOWN_THRESH = 3   # require 3 consecutive connect failures before stop
        UP_THRESH = 2     # require 2 consecutive successes before start
        
        while True:
            if stream_restart_requested.is_set():
                logger.warning("Immediate RTSP restart requested after ffmpeg output failure")
                with encoder_lock:
                    if streaming["running"]:
                        stop_stream()
                    if _rtsp_up(host, port):
                        start_stream()
                stream_restart_requested.clear()
                dead_ticks = 0
                down_ticks = 0
                up_ticks = 0
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
                    if _stream_output_dead():
                        dead_ticks += 1
                        if dead_ticks >= DEAD_THRESH:
                            logger.warning("PyAV output closed for %d checks; restarting stream", dead_ticks)
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
            
            time.sleep(1)

    # Start camera and manager; encoder will start when RTSP is reachable
    with camera_lock:
        picam2.start()
    apply_runtime_camera_settings(config)
    log_camera_timing("Initial startup")
    threading.Thread(target=stream_manager, daemon=True).start()

    w, h = lsize

    with camera_lock:
        prev = picam2.capture_buffer("lores")
    prev = prev[:w * h].reshape(h, w).astype(np.int16)

    bwMode = False # greyscale mode on/off
    with camera_lock:
        picam2.set_controls({"Saturation": 1.0})

    currBrightness = 0
    skipNFrames = 10 # skip the first frames to avoid recording on startup

    while True:
        # Check for configuration changes
        if birdCamera.waitForConfigChange(timeout=0.1):  # Non-blocking check
            logger.info("Configuration changed, reloading...")
            new_config = birdCamera.getCurrentConfig()
            
            # Check if resolution or transform changed (requires camera reconfiguration)
            resolution_changed = (
                config["width"] != new_config["width"] or
                config["height"] != new_config["height"]
            )
            transform_changed = (
                config.get("vflip", 0) != new_config.get("vflip", 0) or
                config.get("hflip", 0) != new_config.get("hflip", 0)
            )

            if resolution_changed or transform_changed:
                if resolution_changed:
                    logger.info(
                        f"Resolution changed from {config['width']}x{config['height']} to "
                        f"{new_config['width']}x{new_config['height']}"
                    )
                if transform_changed:
                    logger.info(
                        f"Flip settings changed to hflip={new_config.get('hflip', 0)}, "
                        f"vflip={new_config.get('vflip', 0)}"
                    )

                reconfigure_camera_pipeline(new_config)
                with camera_lock:
                    prev = picam2.capture_buffer("lores")
                prev = prev[:w * h].reshape(h, w).astype(np.int16)
                skipNFrames = new_config["skippedFramesAfterChange"]
                
            # Update color gains or AWB if they changed (and resolution didn't)
            elif (config["colorOffset_red"] != new_config["colorOffset_red"] or
                  config["colorOffset_blue"] != new_config["colorOffset_blue"] or
                  config.get("awbEnable", False) != new_config.get("awbEnable", False) or
                  config.get("autofocus", False) != new_config.get("autofocus", False) or
                  config.get("framerate", 30) != new_config.get("framerate", 30)):
                try:
                    with camera_lock:
                        if new_config.get("awbEnable", False):
                            # Enable AWB, don't set manual color gains
                            picam2.set_controls({"AwbEnable": True})
                            logger.info("Enabled auto white balance")
                        else:
                            # Disable AWB and set manual color gains
                            picam2.set_controls({
                                "AwbEnable": False,
                                "ColourGains": (
                                    new_config["colorOffset_red"], 
                                    new_config["colorOffset_blue"]
                                )
                            })
                            logger.info("Disabled AWB and set manual color gains")

                    apply_runtime_camera_settings(new_config)
                    config.update(new_config)
                except Exception as e:
                    logger.error(f"Failed to update camera controls: {e}")

                skipNFrames = new_config["skippedFramesAfterChange"]

            else:
                # Other config changes that don't need camera restart
                config.update(new_config)
            
            logger.info(f"Configuration updated: {config}")
            
        # capture new preview and reshape
        with camera_lock:
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
                with camera_lock:
                    picam2.set_controls({"Saturation": 1.0})
                bwMode = False
                skipNFrames = config["skippedFramesAfterChange"]

            elif not bwMode and currBrightness < config["bwSwitchingSensitivity"]:
                logger.info("switching mode to greyscale")
                with camera_lock:
                    picam2.set_controls({"Saturation": 0.0})
                bwMode = True
                skipNFrames = config["skippedFramesAfterChange"]

        prev = cur # overwrite previous frame with current one

if __name__ == "__main__":
    runCamera()
