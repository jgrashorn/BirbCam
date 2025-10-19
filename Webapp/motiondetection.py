import socket
import threading
import subprocess
import os
from datetime import datetime
import time

LISTEN_PORT = 5005
RECORDINGS_DIR = "./recordings"
CAMERA_STREAMS = {
    "garten": "rtsp://localhost:8554/garten",
    # Add more if needed
}

RECORDINGS = {}  # camera_name: subprocess

def start_recording(camera_name):
    # Replace "already recording" check with liveness check
    proc = RECORDINGS.get(camera_name)
    if proc and proc.poll() is None:
        print(f"[{camera_name}] Already recording.")
        return
    elif proc and proc.poll() is not None:
        # Clean up dead process entry
        RECORDINGS.pop(camera_name, None)

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
        "-c:a", "copy",
        "-movflags", "+faststart",
        "-y",
        "-t", "3600",
        output_path
    ], stdin=subprocess.PIPE)
    RECORDINGS[camera_name] = proc

def stop_recording(camera_name):
    proc = RECORDINGS.pop(camera_name, None)
    if proc:
        print(f"[{camera_name}] ⏹️ Stopping recording.")
        try:
            if proc.stdin:
                proc.stdin.write(b"q")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
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
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", LISTEN_PORT))
    sock.listen(5)
    print(f"[Server] Listening on port {LISTEN_PORT}...")

    def reaper():
        # Periodically clean up exited ffmpeg processes
        while True:
            dead = [cam for cam, p in list(RECORDINGS.items()) if p.poll() is not None]
            for cam in dead:
                print(f"[{cam}] Recording process exited.")
                RECORDINGS.pop(cam, None)
            time.sleep(2)

    threading.Thread(target=reaper, daemon=True).start()

    while True:
        conn, addr = sock.accept()
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

if __name__ == "__main__":
    start_server()
