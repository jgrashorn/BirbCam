import time
import datetime
import os
import logging

import numpy as np

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput, FfmpegOutput

import libcamera
import birdCamera

logger = logging.getLogger(__name__)

def runCamera():

    os.environ['PULSE_SERVER'] = "/run/user/1000/pulse/native"

    logging.basicConfig(
                        filename='birb.log',
                        format='%(asctime)s %(levelname)s %(module)s %(funcName)-8s %(message)s',
                        level=logging.INFO,
                        datefmt='%Y-%m-%d %H:%M:%S')
    
    config = birdCamera.readConfig()
    
    lsize = (320, 240) # size of internal preview for motion detection (smol bc fast)
    msize = (config["width"], config["height"]) # size of recording from config.txt
    picam2 = Picamera2()

    video_config = picam2.create_video_configuration(main={"size": msize, "format": "YUV420"}, # format of recording
                                                    lores={"size": lsize, "format": "YUV420"}, # format of preview
                                                    controls={"ColourGains": (0.95, 1.325)} # color correction for IR-cams, (r, b)
                                                    )
    # transforms if camera is not oriented right side up
    video_config["transform"] = libcamera.Transform(hflip=0, vflip=0)
    
    picam2.configure(video_config)
    encoder = H264Encoder(2000000)

    #streamOutput = FfmpegOutput(f'-r 24 -f mpegts udp://localhost:{config["streamPort"]}?pkt_size=1316', audio=True)
    #streamOutput = FfmpegOutput("-f rtsp -rtsp_transport udp rtsp://myuser:mypass@localhost:8554/hqstream", audio=True)
    streamOutput = FfmpegOutput(f'-f rtsp -rtsp_transport udp rtsp://{config["serverIP"]}:{config["rtspPort"]}/{config["name"]}',audio=True, audio_codec="aac", audio_sync=-0.3)
    # streamOutput = FfmpegOutput(f'/home/birb/test.mp4', audio=True, audio_device = 'default')
    # mp4Output = CircularOutput()
    # encoder.output = [streamOutput,mp4Output]
    encoder.output = streamOutput
    picam2.encoders = encoder
    picam2.start()
    picam2.start_encoder()
    
    ltime = time.time()

    # streamOutput.stop()

    w, h = lsize

    encoding = False

    ltime = 0
    starttime = 0
    # fname = ""

    # failedToSend = []

    prev = picam2.capture_buffer("lores")
    prev = prev[:w * h].reshape(h, w).astype(np.int16)

    mseSensitivity = config["sensitivity"]
    pixelThreshold = config["numPixelsThreshold"]
    detectionThreshold = config["detectionThreshold"]

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

        # if no frames are skipped, do the thing
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

            # measure pixels differences (mse) between current and previous frame
            diff = np.subtract(cur, prev)
            mse = np.square(diff).mean()

            # measure individual pixel difference to detect local changes
            diffComp = (np.abs(diff) > detectionThreshold) * np.ones(diff.shape)
        
            if mse > mseSensitivity or np.sum(diffComp) > pixelThreshold:
                
                if not encoding:
                    birdCamera.SendStartTrigger(config["serverIP"],config["port"])
                    logger.info("Started recording.")
                    logger.info( f"New Motion, mse: {mse:5.2f}, diffSum: {np.sum(diffComp):.0f}" )
                    
                    # change thresholds for stopping
                    mseSensitivity = config["stopSensitivity"]
                    pixelThreshold = config["stopNumPixelsThreshold"]
                    detectionThreshold = config["stopDetectionThreshold"]

                    starttime = time.time()
                    encoding = True
                    
                ltime = time.time()
                
            else:
                if encoding and ((time.time() - ltime > config["captureDelay"]) or (time.time() - starttime > config["maxRecordTime"])):

                    birdCamera.SendStopTrigger(config["serverIP"],config["port"])
                    encoding = False
                    logger.info("Stopped.")
                    
                    # change thresholds to starting values
                    mseSensitivity = config["sensitivity"]
                    pixelThreshold = config["numPixelsThreshold"]
                    detectionThreshold = config["detectionThreshold"]
   
        prev = cur # overwrite previous frame with current one

if __name__ == "__main__":
    runCamera()
