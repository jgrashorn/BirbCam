import time
import datetime
import os

import numpy as np

from picamera2 import Picamera2
from picamera2.encoders import H264Encoder, Quality, JpegEncoder
from picamera2.outputs import FileOutput, CircularOutput, FfmpegOutput

#from cv2 import GaussianBlur

import libcamera
import birdCamera
import threading
import socket

def server():
    global circ, picam2
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", 10001))
        sock.listen()
        while tup := sock.accept():
            event = threading.Event()
            conn, addr = tup
            stream = conn.makefile("wb")
            filestream = FileOutput(stream)
            filestream.start()
            encoder.output = [mp4Output, filestream]
            filestream.connectiondead = lambda _: event.set()  # noqa
            event.wait()

config = birdCamera.readConfig()

#NasFolderName = '/media/FritzNAS/Conradfestplatte/Vogelvideos/'
NasFolderName = ''

lsize = (320, 240)
msize = (config["width"], config["height"])
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": msize, "format": "RGB888"},
                                                 lores={"size": lsize, "format": "YUV420"},
                                                 controls={
                                                     "AwbEnable": True,
                                                     "AwbMode": libcamera.controls.AwbModeEnum.Cloudy
                                                    }
                                                )
video_config["transform"] = libcamera.Transform(hflip=1, vflip=1)

picam2.configure(video_config)
#encoder = H264Encoder(1000000,iperiod=60)
encoder = H264Encoder(2000000)

streamOutput = FfmpegOutput(f'-r 30 -f mpegts udp://{config["serverIP"]}:{config["streamPort"]}?pkt_size=1316', audio=False)
#mp4Output = FfmpegOutput("video.mp4", audio=False)
#mp4Output = FileOutput()
mp4Output = CircularOutput()
encoder.output = [streamOutput,mp4Output]
#encoder.output = streamOutput
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

#t = threading.Thread(target=server)
#t.daemon = True
#t.start()

while True:
    
    cur = picam2.capture_buffer("lores")
    cur = cur[:w * h].reshape(h, w).astype(np.int16)
    
    # Measure pixels differences between current and
    # previous frame
    diff = np.subtract(cur, prev)
    
    diffComp = (np.abs(diff) > detectionThreshold) * np.ones(diff.shape)
    
    #diffBlur = np.subtract(GaussianBlur(cur, (5,5), 0), GaussianBlur(prev, (5,5), 0))
    #diffBlurComp = (np.abs(diffBlur) > config["detectionThreshold"]) * np.ones(diffBlur.shape)
    
    mse = np.square(diff).mean()
    
    #if encoding:
        #print("mse: ", mse, "sumdiff: ", np.sum(diffComp), "blurdiff: ", np.sum(diffBlurComp))
        
    if mse > mseSensitivity or np.sum(diffComp) > pixelThreshold:
        
        if not encoding:
            fname = f'Videos/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
            #mp4Output.output_filename = fname
            mp4Output.fileoutput = fname
            mp4Output.start()
            
            mseSensitivity = config["stopSensitivity"]
            pixelThreshold = config["stopNumPixelsThreshold"]
            detectionThreshold = config["stopDetectionThreshold"]

            #np.savetxt(fname + "cur.txt",cur)
            #np.savetxt(fname + "prev.txt",prev)
            #np.savetxt(fname + "diff.txt",diff)
            #np.savetxt(fname + "diffComp.txt",diffComp)
            #np.savetxt(fname + "blur.txt",diffBlur)
            #np.savetxt(fname + "blurComp.txt",diffBlurComp)
            encoding = True
            print("New Motion, mse: ", mse, " diffSum: ", np.sum(diffComp))
        
        ltime = time.time()
        
    else:
        if encoding and ((time.time() - ltime > config["captureDelay"]) or (time.time() - ltime > config["maxRecordTime"])):
            # picam2.stop_encoder()
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
            
            #if not birdCamera.sendVideo(fname + '.mp4',config["serverIP"],config["filePort"]):
            #    failedToSend.append(fname)
            #    print("Failed to send file " + fname + '.mp4')
            #else:
            #    print("Sent!")
            #    cmd ='rm ' + fname + '.mp4'
            #    os.system(cmd)
            #    fname = ""
                
    prev = cur
