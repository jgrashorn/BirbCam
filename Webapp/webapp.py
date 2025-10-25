from flask import Flask, render_template, send_from_directory, request, jsonify, url_for, abort
import os, time, subprocess, socket, json
from pathlib import Path

# Load environment variables from .env file if it exists
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value

# Load environment before importing config
load_env_file()

try:
    from config import config
except ImportError:
    # Try importing from parent directory if running from project root
    import sys
    sys.path.append(str(Path(__file__).parent))
    from config import config

app = Flask(__name__)

# Validate configuration at startup
try:
    config.validate_directories(create=True)
    print(f"✓ Configuration validated for camera: {config.camera}")
    print(f"✓ Media directory: {config.media_dir}")
    print(f"✓ Output directory: {config.output_dir}")
except Exception as e:
    print(f"✗ Configuration error: {e}")
    exit(1)

# Define your available cameras and their HLS paths
cameras = {
    "garten": "http://192.168.178.38:8888/garten/index.m3u8"
    # Add more cameras here
}

# Camera settings endpoints configuration
camera_endpoints = {
    "garten": {"ip": "192.168.178.40", "port": 5005}
    # Add more camera endpoints here
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

@app.route('/settings/<cam>')
def settings(cam):
    if cam not in cameras:
        return "Camera not found", 404
    
    # Get current configuration from camera
    config_response = get_camera_config(cam)
    if config_response.get("status") == "ok":
        current_config = config_response.get("config", {})
    else:
        current_config = {}
        error_message = config_response.get("message", "Failed to get configuration")
    
    return render_template('settings.html', 
                         cam=cam, 
                         config=current_config,
                         error=config_response.get("message") if config_response.get("status") != "ok" else None)

@app.route('/api/settings/<cam>', methods=['GET'])
def api_get_settings(cam):
    if cam not in cameras:
        return jsonify({"status": "error", "message": "Camera not found"}), 404
    
    return jsonify(get_camera_config(cam))

@app.route('/api/settings/<cam>', methods=['POST'])
def api_set_settings(cam):
    if cam not in cameras:
        return jsonify({"status": "error", "message": "Camera not found"}), 404
    
    try:
        new_config = request.get_json()
        if not new_config:
            return jsonify({"status": "error", "message": "No configuration provided"}), 400
        
        result = set_camera_config(cam, new_config)
        
        if result.get("status") == "ok":
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/ping/<cam>')
def api_ping_camera(cam):
    if cam not in cameras:
        return jsonify({"status": "error", "message": "Camera not found"}), 404
    
    return jsonify(ping_camera(cam))

# Use centralized configuration
RECORD_ROOT = config.media_dir
WEB_RECORD_ROOT = config.output_dir

def _ffprobe_duration(path: Path) -> float:
    try:
        out = subprocess.check_output(
            ["ffprobe","-v","error","-show_entries","format=duration","-of","default=nk=1:nw=1", str(path)],
            text=True
        )
        return float(out.strip())
    except Exception:
        return 0.0

def _trim(src: Path, start: float, end: float, dst: Path) -> bool:
    dur = max(0.0, end - start)
    if dur <= 0.05:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    # try copy
    cmd = ["ffmpeg","-hide_banner","-loglevel","error",
           "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
           "-map","0:v:0?","-map","0:a:0?","-c:v","copy","-c:a","copy",
           "-movflags","+faststart","-y", str(dst)]
    if subprocess.call(cmd) == 0 and dst.exists() and dst.stat().st_size > 0:
        return True
    # re-encode fallback
    cmd = ["ffmpeg","-hide_banner","-loglevel","error",
           "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
           "-map","0:v:0?","-map","0:a:0?","-c:v","libx264","-preset","veryfast","-crf","20",
           "-c:a","aac","-b:a","128k","-ac","2","-ar","48000",
           "-movflags","+faststart","-shortest","-y", str(dst)]
    return subprocess.call(cmd) == 0 and dst.exists() and dst.stat().st_size > 0

def _concat(parts: list[Path], dst: Path) -> bool:
    lst = dst.with_suffix(".txt")
    with lst.open("w") as f:
        for p in parts:
            f.write(f"file '{p.as_posix()}'\n")
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg","-hide_banner","-loglevel","error",
           "-f","concat","-safe","0","-i", str(lst),
           "-c","copy","-movflags","+faststart","-y", str(dst)]
    rc = subprocess.call(cmd)
    try: lst.unlink(missing_ok=True)
    except Exception: pass
    if rc == 0 and dst.exists() and dst.stat().st_size > 0:
        return True
    # fallback re-encode
    with lst.open("w") as f:
        for p in parts:
            f.write(f"file '{p.as_posix()}'\n")
    cmd = ["ffmpeg","-hide_banner","-loglevel","error",
           "-f","concat","-safe","0","-i", str(lst),
           "-c:v","libx264","-preset","veryfast","-crf","20",
           "-c:a","aac","-b:a","128k","-movflags","+faststart","-y", str(dst)]
    rc = subprocess.call(cmd)
    try: lst.unlink(missing_ok=True)
    except Exception: pass
    return rc == 0 and dst.exists() and dst.stat().st_size > 0

def communicate_with_camera(cam_name: str, command: str, timeout=10):
    """Send command to camera settings server and get response."""
    if cam_name not in camera_endpoints:
        return {"status": "error", "message": f"Unknown camera: {cam_name}"}
    
    endpoint = camera_endpoints[cam_name]
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect((endpoint["ip"], endpoint["port"]))
            
            # Send command
            sock.send((command + "\n").encode('utf-8'))
            
            # Receive response
            response = sock.recv(4096).decode('utf-8').strip()
            
            return json.loads(response)
            
    except socket.timeout:
        return {"status": "error", "message": "Connection timeout"}
    except ConnectionRefusedError:
        return {"status": "error", "message": "Camera not reachable"}
    except json.JSONDecodeError:
        return {"status": "error", "message": "Invalid response from camera"}
    except Exception as e:
        return {"status": "error", "message": f"Communication error: {str(e)}"}

def get_camera_config(cam_name: str):
    """Get current configuration from camera."""
    return communicate_with_camera(cam_name, "GET_CONFIG")

def set_camera_config(cam_name: str, config: dict):
    """Send new configuration to camera."""
    config_json = json.dumps(config)
    return communicate_with_camera(cam_name, f"SET_CONFIG:{config_json}")

def ping_camera(cam_name: str):
    """Ping camera to check if it's reachable."""
    return communicate_with_camera(cam_name, "PING")

@app.post("/api/clip")
def api_clip_last():
    data = request.get_json(silent=True) or {}
    cam = data.get("cam")
    seconds = int(data.get("seconds", 60))
    if not cam:
        return jsonify(ok=False, error="Missing cam"), 400

    cam_dir = (RECORD_ROOT / cam)
    if not cam_dir.exists():
        return jsonify(ok=False, error="No recordings for camera"), 404

    files = sorted(cam_dir.rglob("*.mp4"), key=lambda p: (p.stat().st_mtime, str(p)))
    if not files:
        return jsonify(ok=False, error="No files"), 404

    last = files[-1]
    last_dur = _ffprobe_duration(last)
    need = seconds

    prev = files[-2] if len(files) > 1 else None
    prev_dur = _ffprobe_duration(prev) if prev else 0.0

    # Figure trim ranges (FIX: define take_last, t1_start)
    take_last = min(need, last_dur)
    t1_start = max(0.0, last_dur - take_last)

    parts: list[Path] = []
    tmp_parts: list[Path] = []
    stamp = int(time.time())
    day = time.strftime("%Y-%m-%d", time.localtime(last.stat().st_mtime))
    out_dir = (WEB_RECORD_ROOT / cam / day)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{cam}_{stamp}_last{seconds}s.mp4"
    out_path = out_dir / out_name
    # temp files stay alongside final clip
    tmp1 = out_dir / f"._tmp_last_{stamp}_a.mp4"
    if not _trim(last, t1_start, last_dur, tmp1):
        return jsonify(ok=False, error="Trim failed (last)"), 500
    parts.append(tmp1); tmp_parts.append(tmp1)
    need -= take_last

    # If we still need more, take tail of prev
    if need > 0 and prev and prev_dur > 0:
        t0_start = max(0.0, prev_dur - need)
        tmp0 = out_dir / f"._tmp_last_{stamp}_b.mp4"
        if not _trim(prev, t0_start, prev_dur, tmp0):
            for t in tmp_parts:
                try: t.unlink(missing_ok=True)
                except Exception: pass
            return jsonify(ok=False, error="Trim failed (prev)"), 500
        parts.insert(0, tmp0); tmp_parts.append(tmp0)

    # Single part -> move/rename; else concat
    if len(parts) == 1:
        parts[0].rename(out_path)
    else:
        if not _concat(parts, out_path):
            for t in tmp_parts:
                try: t.unlink(missing_ok=True)
                except Exception: pass
            return jsonify(ok=False, error="Concat failed"), 500

    # Cleanup temps
    for t in tmp_parts:
        try: t.unlink(missing_ok=True)
        except Exception: pass

    try:
        rel_url = f"{day}/{out_name}"
        url = url_for("serve_recording", cam=cam, relpath=rel_url)
    except Exception:
        url = f"/recordings/{cam}/{out_name}"

    return jsonify(ok=True, filename=out_name, url=url)

def _organize_recordings(cam_dir: Path):
    """Move loose MP4s into YYYY-MM-DD subfolders."""
    for mp4 in cam_dir.glob("*.mp4"):
        day = time.strftime("%Y-%m-%d", time.localtime(mp4.stat().st_mtime))
        day_dir = cam_dir / day
        day_dir.mkdir(parents=True, exist_ok=True)
        target = day_dir / mp4.name
        if target.exists():
            continue
        mp4.rename(target)

@app.route('/recordings/<cam>')
def recordings(cam):
    cam_dir = WEB_RECORD_ROOT / cam
    if not cam_dir.exists():
        return f"No recordings found for {cam}", 404

    _organize_recordings(cam_dir)

    groups = []
    for day_dir in sorted([d for d in cam_dir.iterdir() if d.is_dir()],
                          key=lambda d: d.name, reverse=True):
        files = sorted(day_dir.glob("*.mp4"),
                       key=lambda p: p.stat().st_mtime,
                       reverse=True)
        groups.append({
            "date": day_dir.name,
            "files": [{"name": f.name,
                       "rel": f"{day_dir.name}/{f.name}",
                       "ts": f.stat().st_mtime} for f in files]
        })

    # pick selected file (from query ?rel= or latest)
    selected_rel = request.args.get("rel")
    if not selected_rel:
        # choose newest day, newest file
        if groups and groups[0]["files"]:
            selected_rel = groups[0]["files"][0]["rel"]
    selected_url = url_for("serve_recording", cam=cam, relpath=selected_rel) if selected_rel else None
    selected_name = selected_rel.split("/")[-1] if selected_rel else None

    return render_template('recordings.html', cam=cam, groups=groups,
                           selected_url=selected_url, selected_name=selected_name)

@app.route('/recordings/<cam>/<date>')
def recordings_day(cam, date):
    cam_dir = WEB_RECORD_ROOT / cam
    day_dir = cam_dir / date
    if not day_dir.exists():
        return f"No recordings for {cam} on {date}", 404

    files = sorted(day_dir.glob("*.mp4"),
                   key=lambda p: p.stat().st_mtime,
                   reverse=True)
    if not files:
        return f"No recordings for {cam} on {date}", 404

    data = [{"name": f.name,
             "rel": f"{day_dir.name}/{f.name}",
             "ts": f.stat().st_mtime} for f in files]

    # default selected: first file of the day
    selected_rel = data[0]["rel"]
    selected_url = url_for("serve_recording", cam=cam, relpath=selected_rel)

    return render_template('recordings.html', cam=cam, groups=[{"date": date, "files": data}],
                           selected_url=selected_url, selected_name=data[0]["name"])

@app.route('/recordings/<cam>/<path:relpath>')
def serve_recording(cam, relpath):
    cam_dir = WEB_RECORD_ROOT / cam
    if not cam_dir.exists():
        return f"No recordings for {cam}", 404
    requested = (cam_dir / relpath).resolve()
    if not str(requested).startswith(str(cam_dir.resolve())):
        abort(403)
    if not requested.exists():
        return "File not found", 404
    return send_from_directory(str(cam_dir), relpath)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

