from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

# Define your available cameras and their HLS paths
cameras = {
    "garten": "http://192.168.178.38:8888/garten/index.m3u8"
    # Add more cameras here
}

@app.route('/')
def index():
    return render_template('index.html', cameras=cameras)

@app.route('/camera/<name>')
def camera(name):
    stream_url = cameras.get(name)
    if not stream_url:
        return "Camera not found", 404
    return render_template('camera.html', name=name, stream_url=stream_url)

@app.route('/recordings/<cam>')
def recordings(cam):
    path = f"recordings/{cam}"
    if not os.path.exists(path):
        return f"No recordings found for {cam}", 404
    files = sorted(os.listdir(path), reverse=True)
    return render_template('recordings.html', cam=cam, files=files)

@app.route('/recordings/<cam>/<filename>')
def serve_recording(cam, filename):
    return send_from_directory(f"recordings/{cam}", filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

