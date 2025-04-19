Python script for a bird camera with automatic movement detection, recording and streaming.

# Useful

## Install picamera2
https://pypi.org/project/picamera2/

## picamera2 in virtual environment
`python3 -m venv --system-site-packages env`

## Source for the AI-stuff
https://pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/

## Source for the Webserver-stuff
https://pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/

## systemd-service
```
[Unit]
Description=birbCam

[Service]
WorkingDirectory=</path/to/folder>/Camera/
User=<username>
ExecStart=</path/to/folder>/.venv/bin/python </path/to/folder>/Camera/cameraPython.py
```

## mediamtx
```
https://github.com/bluenviron/mediamtx
```
