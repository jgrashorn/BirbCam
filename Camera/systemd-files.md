# Camera

Service to run the camera.

```
[Unit]
Description=birbCamera
Wants=network-online.target sound.target
After=network-online.target sound.target

[Service]
User=birb
Group=birb
WorkingDirectory=/home/birb/BirbCam/Camera
ExecStart=/home/birb/BirbCam/Camera/.venv/bin/python /home/birb/BirbCam/Camera/cameraPython.py
Restart=always
RestartSec=2
StandardOutput=journal
StandardError=journal
KillMode=control-group

[Install]
WantedBy=multi-user.target
```

