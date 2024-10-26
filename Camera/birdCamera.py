import json

import socket
import os
import struct

import io
import logging
import threading

from picamera2.outputs import FileOutput

PAGE ="""\
            <html>
            <head>
            <title>picamera2 MJPEG streaming demo</title>
            </head>
            <body>
            <h1>Picamera2 MJPEG Streaming Demo</h1>
            <img src="stream.mjpg" width="640" height="480" />
            </body>
            </html>
            """

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
    
    print("Connecting to " + ip + " on port " + f"{port}")
    print("Sending file "+ filename)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    # Connecting with Server 
    sock.connect((ip, port)) 
    
    print("Connected!")
    
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
        print("IOError")
                
    return success