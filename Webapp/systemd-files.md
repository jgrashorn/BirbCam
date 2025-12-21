# Webapp

Service to run the webapp

## birbServer.service

```
[Unit]
Description=birbServer
After=network.target

[Service]
WorkingDirectory=/home/birb/BirbCam/Webapp/
User=birb
ExecStart=/home/birb/BirbCam/Webapp/.venv/bin/python /home/birb/BirbCam/Webapp/webapp.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

# Mediamtx

Service to run mediamtx

## mediamtx.service

```
[Unit]
Description=mediamtx
After=network.target

[Service]
WorkingDirectory=/home/birb/mediamtx
User=birb
ExecStart=/home/birb/mediamtx/mediamtx
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

# Motion detection

Service and timer to run the motion detection periodically (`motiondetection.py`)

## birbcam-motion.timer

```
[Unit]
Description=Run BirbCam motion extraction with 5-minute gap

[Timer]
OnBootSec=1min
OnUnitActiveSec=5min
Persistent=true
AccuracySec=30s
Unit=birbcam-motion.service

[Install]
WantedBy=timers.target
```

## birbcam-motion.service
```
[Unit]
Description=BirbCam motion extraction

[Service]
Type=oneshot
User=birb
Group=birb
WorkingDirectory=/home/birb/BirbCam/Webapp/
ExecStart=/home/birb/BirbCam/Webapp/.venv/bin/python /home/birb/BirbCam/Webapp/motiondetection.py
```

# Recording and processing in system memory

Services to create and mount tmfps directories for processing in system memory. Could help with speed and possibly also reduce SD-card wear, if SD-card is used.

## mnt-birbtmp.mount

Mount tmpfs directory

```
[Unit]
Description=Temporary in-memory filesystem at /mnt/birbtmp
DefaultDependencies=no
Before=local-fs.target
After=swap.target

[Mount]
What=tmpfs
Where=/mnt/birbtmp
Type=tmpfs
Options=mode=1777,size=1000M

[Install]
WantedBy=multi-user.target
```

## mnt-birbtmp-setup.service

Create subdirectories and setup permissions

```
[Unit]
Description=Create subdirectories inside /mnt/birbtmp
After=mnt-birbtmp.mount
Requires=mnt-birbtmp.mount

[Service]
Type=oneshot
ExecStart=/usr/bin/mkdir -p /mnt/birbtmp/mediamtx /mnt/birbtmp/temp
ExecStart=/usr/bin/chown -R birb:birb /mnt/birbtmp/mediamtx /mnt/birbtmp/temp
ExecStart=/usr/bin/chmod 755 /mnt/birbtmp/mediamtx /mnt/birbtmp/temp

[Install]
WantedBy=multi-user.target
```
