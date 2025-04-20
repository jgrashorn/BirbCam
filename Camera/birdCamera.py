import json

import socket
import os
import struct
import logging

import io
import logging
import threading

from picamera2.outputs import FileOutput

logger = logging.getLogger(__name__)

def readConfig():
    with open("config.txt") as f:
        configData = f.read()

    parsedConfig = json.loads(configData)
    
    return parsedConfig
    
def send_file(sck: socket.socket, filename):
    # Get the size of the outgoing file.
    filesize = os.path.getsize(filename)
    # First inform the server the amount of
    # bytes that will be sent.
    sck.sendall(struct.pack("<Q", filesize))
    # Send the file in 1024-bytes chunks.
    with open(filename, "rb") as f:
        while read_bytes := f.read(1024):
            sck.sendall(read_bytes)


def sendVideo(filename,ip,port):
    
    logger.info("Connecting to " + ip + " on port " + f"{port}")
    logger.info("Sending file "+ filename)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    # Connecting with Server 
    sock.connect((ip, port)) 
    
    logger.info("Connected!")
    
    success = False
    try:
        sock.send(filename.encode())
        returnMsg = sock.recv(1024).decode()
        if returnMsg == "OK":
            send_file(sock, filename)
        else:
            raise(IOError)
        success = True
  
    except IOError: 
        logger.warning("IOError")
                
    return success
    
def SendStartTrigger(ip, port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((ip, port))
         
        sock.send(b"garten start\n")
    except IOError:
        logger.warning("IOError")
         
def SendStopTrigger(ip, port):
     
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((ip, port))
         
        sock.send(b"garten stop\n")
    except IOError:
        logger.warning("IOError")
<<<<<<< HEAD
     
=======
     
>>>>>>> 03d5fc44a9db5ab76be4e46e48d5871b4f28b69a
