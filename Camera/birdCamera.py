import json
import logging

from picamera2.outputs import FileOutput

logger = logging.getLogger(__name__)

def readConfig():
    with open("config.txt") as f:
        configData = f.read()

    parsedConfig = json.loads(configData)
    
    return parsedConfig

