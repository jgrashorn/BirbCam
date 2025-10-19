from flask import Flask, render_template, send_from_directory, request, jsonify, url_for
import os, time, subprocess
from pathlib import Path

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

RECORD_ROOT = Path(os.environ.get("BIRBCAM_MEDIA_DIR", "/home/birb/mediamtx/recordings"))
WEB_RECORD_ROOT = Path(os.environ.get("BIRBCAM_OUTPUT_DIR", "/home/birb/BirbCam/Webapp/recordings"))

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

    # Figure trim ranges
    parts: list[Path] = []
    tmp_parts: list[Path] = []
    stamp = int(time.time())
    out_dir = WEB_RECORD_ROOT / cam
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"{cam}_{stamp}_last{seconds}s.mp4"
    out_path = out_dir / out_name

    # Always include tail of last file
    take_last = min(need, last_dur)
    t1_start = max(0.0, last_dur - take_last)
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

    # If you have a serve_recording route, build its URL; else return static path
    try:
        url = url_for("serve_recording", cam=cam, filename=out_name)
    except Exception:
        url = f"/recordings/{cam}/{out_name}"

    return jsonify(ok=True, filename=out_name, url=url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

