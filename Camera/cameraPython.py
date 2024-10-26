import time
import datetime
import os

import numpy as np

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput, FfmpegOutput

import libcamera
import birdCamera

config = birdCamera.readConfig()

NasFolderName = '' # placeholder if videos should be stored in some other folder (e.g. a NAS)

lsize = (320, 240)
msize = (config["width"], config["height"])
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": msize, "format": "YUV420"},
                                                 lores={"size": lsize, "format": "YUV420"},
                                                 controls={"ColourGains": (1.0, 1.325)}
                                                 )
# transforms if camera is not oriented right side up
video_config["transform"] = libcamera.Transform(hflip=1, vflip=1)

picam2.configure(video_config)
encoder = H264Encoder(2000000)

streamOutput = FfmpegOutput(f'-r 30 -f mpegts udp://{config["serverIP"]}:{config["streamPort"]}?pkt_size=1316', audio=False)
mp4Output = CircularOutput()
encoder.output = [streamOutput,mp4Output]
picam2.encoders = encoder
picam2.start()
picam2.start_encoder()

w, h = lsize

encoding = False

ltime = 0
fname = ""

failedToSend = []

prev = picam2.capture_buffer("lores")
prev = prev[:w * h].reshape(h, w).astype(np.int16)

mseSensitivity = config["sensitivity"]
pixelThreshold = config["numPixelsThreshold"]
detectionThreshold = config["detectionThreshold"]

while True:
    
    cur = picam2.capture_buffer("lores")
    cur = cur[:w * h].reshape(h, w).astype(np.int16)
    
    # Measure pixels differences between current and
    # previous frame
    diff = np.subtract(cur, prev)
    
    diffComp = (np.abs(diff) > detectionThreshold) * np.ones(diff.shape)
    
    mse = np.square(diff).mean()
        
    if mse > mseSensitivity or np.sum(diffComp) > pixelThreshold:
        
        if not encoding:
            fname = f'Videos/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            mp4Output.fileoutput = fname
            mp4Output.start()
            
            mseSensitivity = config["stopSensitivity"]
            pixelThreshold = config["stopNumPixelsThreshold"]
            detectionThreshold = config["stopDetectionThreshold"]

            encoding = True
            # print("New Motion, mse: ", mse, " diffSum: ", np.sum(diffComp))
        
        ltime = time.time()
        
    else:
        if encoding and ((time.time() - ltime > config["captureDelay"]) or (time.time() - ltime > config["maxRecordTime"])):

            mp4Output.stop()
            encoding = False
            print("Stopped.")
            
            mseSensitivity = config["sensitivity"]
            pixelThreshold = config["numPixelsThreshold"]
            detectionThreshold = config["detectionThreshold"]
            
            cmd = 'ffmpeg -nostats -loglevel 0 -r 30 -i ' + fname + ' -c copy ' + NasFolderName + fname +'.mp4'
            os.system(cmd)
            
            cmd ='rm ' + fname
            os.system(cmd)
            
            # some code to send video to network-server

            #if not birdCamera.sendVideo(fname + '.mp4',config["serverIP"],config["filePort"]):
            #    failedToSend.append(fname)
            #    print("Failed to send file " + fname + '.mp4')
            #else:
            #    print("Sent!")
            #    cmd ='rm ' + fname + '.mp4'
            #    os.system(cmd)
            #    fname = ""
                
    prev = cur
