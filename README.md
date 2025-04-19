Python script for a bird camera with automatic movement detection, recording and streaming. Sort of stable and ugly af right now, but seems to be working somehow (thanks ChatGPT!).
After trying for weeks with ffmpeg, libcamera, picamera, frigate and whatever else Google suggested to get something coherent that can live stream and record on events with video AND audio I settled for this POS.

# Client (Camera)

Client streams video and audio via an ffmpeg rtsp stream to the server. Also checks for motion on a 

## Install picamera2
https://pypi.org/project/picamera2/

## picamera2 in virtual environment
`python3 -m venv --system-site-packages .venv`

## config
rename config_default.txt, change Server address, port, sensitivities etc.

## systemd-service
```
[Unit]
Description=birbCam

[Service]
WorkingDirectory=</path/to/folder>/Camera/
User=<username>
ExecStart=</path/to/folder>/.venv/bin/python </path/to/folder>/Camera/cameraPython.py
```

# Server

## mediamtx

For rtsp-Server
```
https://github.com/bluenviron/mediamtx
```

## webapp
```
multi_cam_dashboard/
├── app.py
├── templates/
│   ├── index.html
│   └── camera.html
├── static/
│   └── (optional: style.css, logos, etc.)
```

# Useful

## Source for the AI-stuff
https://pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/

## Source for the Webserver-stuff
https://pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/
