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

def start_server(host='0.0.0.0', port=65432):
    # Create a TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)  # Listen for incoming connections, allowing 1 connection at a time
    print(f"Server started and listening on {host}:{port}")

    try:
        while True:
            # Accept a connection from the client
            client_socket, client_address = server_socket.accept()
            print(f"Connected to client at {client_address}")

            # Receive data in chunks of 1024 bytes
            data = client_socket.recv(1024).decode('utf-8')
            if data:
                print(f"Received command: {data}")
                # You could add code here to handle different commands as needed

            # Close the client connection
            client_socket.close()
            print("Connection closed")
    except KeyboardInterrupt:
        print("Server shutting down.")
    finally:
        # Clean up the server socket
        server_socket.close()