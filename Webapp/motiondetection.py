import socket
import threading
import subprocess
import os
from datetime import datetime

LISTEN_PORT = 5005
RECORDINGS_DIR = "./recordings"
CAMERA_STREAMS = {
    "garten": "rtsp://localhost:8554/hqstream",
    # Add more if needed
}

RECORDINGS = {}  # camera_name: subprocess

def start_recording(camera_name):
    if camera_name in RECORDINGS:
        print(f"[{camera_name}] Already recording.")
        return

    stream_url = CAMERA_STREAMS.get(camera_name)
    if not stream_url:
        print(f"[{camera_name}] Unknown camera.")
        return

    os.makedirs(os.path.join(RECORDINGS_DIR, camera_name), exist_ok=True)
    filename = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(RECORDINGS_DIR, camera_name, f"{filename}.mp4")

    print(f"[{camera_name}] ▶️ Starting recording → {output_path}")
    proc = subprocess.Popen([
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", stream_url,
        "-c:v", "copy",
        "-c:a", "aac",
        "-y",
        "-t", "3600",  # max duration, you’ll kill it early
        output_path
    ])
    RECORDINGS[camera_name] = proc

def stop_recording(camera_name):
    proc = RECORDINGS.pop(camera_name, None)
    if proc:
        print(f"[{camera_name}] ⏹️ Stopping recording.")
        proc.terminate()
    else:
        print(f"[{camera_name}] Not recording.")

def handle_client(conn, addr):
    try:
        data = conn.recv(1024).decode("utf-8").strip()
        print(f"[TCP] Received: {data} from {addr}")
        if data:
            parts = data.split()
            if len(parts) == 2:
                cam, action = parts
                if action == "start":
                    start_recording(cam)
                elif action == "stop":
                    stop_recording(cam)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

def start_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("0.0.0.0", LISTEN_PORT))
    sock.listen(5)
    print(f"[Server] Listening on port {LISTEN_PORT}...")

    while True:
        conn, addr = sock.accept()
        threading.Thread(target=handle_client, args=(conn, addr)).start()

if __name__ == "__main__":
    start_server()
